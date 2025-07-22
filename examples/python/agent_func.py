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
    reward = 0

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

        info = await match.adjudicate({'player_0': answer})
        previous_play = match.players[1].history[-1] if match and match.players[1].history else None
        scores = info.get("scores", None) if match else None
        
        next_observation = f"""In the previous round (round {idx}), the other player chose {previous_play}. 
        The score for this round is {scores[-1] if scores else 0}, 
        the cumulative score is {match.final_score() if match else 0}."""

        reward = scores[-1]
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON observation: {e}")
        # Handle fallback case
        next_observation="Failed to parse JSON observation. Please output in the right JSON format."
        reward = -5
        
    except ValueError as e:
        print(f"Invalid response format: {e}")
        # Handle validation errors
        next_observation="Invalid response format. Please output in the right JSON format"
        reward = -5
    
    done = False # pretty useless in this code but may be useful later. keep.

    return {
        "rewards": reward,  # Rewards for advantage calculation
        "scores": reward,  # Scores for dynamic filtering (0-1 reward)
        "next_observation": next_observation,  # The updated observation for vLLM inference in next step
        "done": done,  # Boolean indicating if the episode is complete
        "extra_logs": {"dummy_scores": reward, "reasoning": reasoning, "parsed_answer": answer},  # Additional logging information
        "match": match,
    }
