import asyncio
import os

# Add Axelrod to the path
import sys
sys.path.append("/net/scratch2/machiavellm/Axelrod/axelrod")
import axelrod as axl

import ray

from .vllm_engine import BaseLLMRayActor


@ray.remote
class AgentInstance:
    def __init__(self, agent_func_path):
        if agent_func_path.endswith(".py"):
            import importlib.util

            spec = importlib.util.spec_from_file_location("step", agent_func_path)
            agent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent_module)
            self.agent_step = agent_module.step
        else:
            raise ValueError("Agent path must be a Python file")

    async def step(self, observation, action, label, **kwargs):
        return await self.agent_step(observation, action, label, **kwargs)


@ray.remote
def get_tokenize_text_len(text, tokenizer):
    return len(tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0])


@ray.remote
class LLMRayActorAsync(BaseLLMRayActor):
    async def __init__(self, *args, bundle_indices: list = None, **kwargs):
        self.agent_func_path = kwargs.pop("agent_func_path")

        # Initialize super class
        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        # Initialize result queue for streaming completed results
        self.result_queue = asyncio.Queue()

        os.environ["VLLM_USE_V1"] = "1"
        import vllm

        assert vllm.__version__ > "0.8.5", "Asyn VLLM version must be greater than 0.8.5"

        engine_args = vllm.AsyncEngineArgs(*args, **self.kwargs)
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        await self.llm.is_sleeping()

    async def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray
    ):
        return await self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    async def update_weight(self, name, dtype, shape, empty_cache=False):
        return await self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    async def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return await self.llm.collective_rpc(
            "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
        )

    async def reset_prefix_cache(self):
        await self.llm.reset_prefix_cache()

    async def sleep(self, level=1):
        await self.llm.sleep(level=level)

    async def wake_up(self):
        await self.llm.wake_up()

    async def add_requests(self, sampling_params, prompts, labels, max_length, hf_tokenizer=None, max_steps=10000):
        """
        Process requests from rank0 and generate responses with multiple agent interactions.
        Each prompt will go through multiple steps of interaction using the step function.
        Results are streamed back as each agent completes its execution.

        Args:
            sampling_params: Parameters for sampling
            prompts: List of prompts to process
            labels: List of labels corresponding to prompts
            max_steps: Maximum number of interaction steps
        """

        # Create semaphore to control concurrent task execution
        NUM_TASKS = os.environ.get("OPENRLHF_ASYNC_NUM_TASKS", 128)
        semaphore = asyncio.Semaphore(NUM_TASKS)

        async def execute_agent(prompt, label, sampling_params):
            async with semaphore:
                # Create a unique agent instance for this prompt
                agent_instance = AgentInstance.remote(self.agent_func_path)

                # Initialize observations and actions for the current prompt
                observation = prompt
                action_ranges = []
                total_reward = 0
                final_scores = 0
                max_steps = 10

                # Initialize Axelrod players
                import sys
                import time
                start = time.time()
                sys.path.append("/net/scratch2/machiavellm/Axelrod/axelrod")

                import asyncio
                from axelrod.strategies.grudger import ForgetfulGrudger
                from axelrod.strategies.lookerup import EvolvedLookerUp2_2_2
                from axelrod.strategies.finite_state_machines import EvolvedFSM16
                from axelrod.strategies.hmm import EvolvedHMM5
                from axelrod.strategies.defector   import Defector
                from axelrod.strategies.human import Human
                from axelrod.match import AsyncMatch
                from axelrod.game import Game
                print(f"Import took: {time.time() - start:.2f} seconds")

                game = axl.Game(r=3, s=-2, t=5, p=0) # this one is very tempting
                noise_level = 0.05
    
                match = axl.AsyncMatch(
                    [Human(), EvolvedHMM5()], 
                    turns=max_steps, 
                    game=game,
                    noise=noise_level,  # Add noise to the match
                    seed=42  # Add seed for reproducibility
                )

                await match.start_match()

                # Execute multiple steps of interaction
                for step_idx in range(max_steps):
                    # Next sampling budget
                    observation_tokens_len = len(
                        hf_tokenizer(observation, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                    )
                    sampling_params.max_tokens = max_length - observation_tokens_len
                    # No budget to generate, break
                    if sampling_params.max_tokens <= 0:
                        break

                    # Generate response asynchronously
                    request_output = await self.generate_async(observation, sampling_params)
                    action = request_output.outputs[0].text
                    action_ranges.append((len(observation), len(observation) + len(action)))

                    # Call step function to get reward and next observation
                    # Use asyncio.to_thread to make Ray remote call non-blocking
                    kwargs = {"sampling_params": sampling_params, 
                    "max_rounds": max_steps, 
                    "step_idx": step_idx,
                    "match": match}

                    result = await agent_instance.step.remote(observation, action, label, **kwargs)
                    total_reward += result["rewards"].item()
                    final_scores = result.get("scores", total_reward)
                    observation = result["next_observation"]
                    done = result["done"]
                    extra_logs = result.get("extra_logs", {})
                    
                    # Update match
                    match = result.get("match", None)

                    # Get sampling params from the environment step
                    if result.get("sampling_params", None):
                        sampling_params = result["sampling_params"]

                    if done:
                        break
                print(f"Match final score: {match.final_score()}")
                print(match.sparklines())
                ray.kill(agent_instance)

                # Store the final response when agent execution is complete
                final_response = {
                    "prompt": prompt,
                    "label": label,
                    "observation": observation,
                    "reward": total_reward,
                    "scores": final_scores,
                    "extra_logs": extra_logs,
                    "action_ranges": action_ranges,
                }
                await self.result_queue.put(final_response)

        # Create and start tasks for all agent executions with controlled concurrency
        import copy

        tasks = []
        for prompt, label in zip(prompts, labels):
            tasks.append(execute_agent(prompt, label, copy.deepcopy(sampling_params)))

        # Run the async code using the class's event loop
        await asyncio.gather(*tasks)

    async def generate_async(self, prompts, sampling_params):
        from vllm.utils import random_uuid

        request_id = random_uuid()
        results_generator = self.llm.generate(prompts, sampling_params, request_id)
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        return final_output

    async def get_responses(self):
        """
        Synchronously get all completed agent results from the queue.
        Waits for all tasks to complete before returning results.
        Returns: List of all completed agent results.
        """
        # Get all results from the queue
        results = []
        while not self.result_queue.empty():
            try:
                results.append(await self.result_queue.get())
            except asyncio.QueueEmpty:
                break
        return results
