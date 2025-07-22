import random
from typing import Any, Dict

import torch
import time
import os
import sys
from typing import Dict, Any
from pydantic import BaseModel
import json

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

class IteratedPrisonerDilemmaResponse(BaseModel):
    reasoning: str
    answer: str


# A n-step random environment
async def step(observation, action, label, **kwargs) -> Dict[str, Any]:
    
    match = kwargs.get("match", None)
    idx = kwargs.get("step_idx", 0)

    try:
        if isinstance(observation, str):
            observation_data = json.loads(observation)
        else:
            observation_data = observation
            
        response = IteratedPrisonerDilemmaResponse(**observation_data)
        reasoning = response.reasoning
        answer = response.answer  
        
        print(f"Agent reasoning: {reasoning}")
        print(f"Agent action: {answer}")
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON observation: {e}")
        # Handle fallback case
        answer = "C"  # default action
        reasoning = "Failed to parse response"
        
    except ValueError as e:
        print(f"Invalid response format: {e}")
        # Handle validation errors
        answer = "C"  # default action
        reasoning = "Invalid response format"

    if match is not None:
        info = await match.adjudicate({'player_0': answer})

    previous_play = match.players[1].history[-1] if match.players[1].history else None
    scores = info.get("scores", None) if match else None
    
    next_observation = f"In the previous round (round {idx}), the other player chose {previous_play}. The score for this round is {scores[-1]}, the cumulative score is {match.final_score()}."
    reward = 1.0

    return {
        "rewards": reward,  # Rewards for advantage calculation
        "scores": reward,  # Scores for dynamic filtering (0-1 reward)
        "next_observation": next_observation,  # The updated observation for vLLM inference in next step
        "done": done,  # Boolean indicating if the episode is complete
        "sampling_params": kwargs.get("sampling_params", None),  # Parameters for vLLM sampling in next step
        "extra_logs": {"dummy_scores": reward, "reasoning": reasoning, "parsed_answer": answer},  # Additional logging information
        "match": match,
    }
