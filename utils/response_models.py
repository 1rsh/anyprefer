from typing import Tuple
from pydantic import BaseModel, Field

class JudgeResponse(BaseModel):
    reasoning: str = Field(..., description="The reasoning behind the score assigned to the response.")
    score: int = Field(..., ge=1, le=10, description="Score assigned to the response on a scale of 1 to 10.")

class RewardResponse(BaseModel):
    analysis: str = Field(..., description="The analysis of the data pair.")
    score: int = Field(..., description="The score assigned to the data pair.")

class PromptOptimizationResponse(BaseModel):
    optimized_target_system_prompt: str = Field(..., description="The optimized target system prompt.")
    optimized_judge_system_prompt: str = Field(..., description="The optimized judge system prompt.")

class PreferenceData(BaseModel):
    prompt: str
    positive_response: str
    negative_response: str
    details: Tuple[Tuple[JudgeResponse, JudgeResponse], RewardResponse, str, str]

    def __str__(self):
        return f"PreferenceData(prompt={self.prompt}, positive_response={self.positive_response[:100]}..., negative_response={self.negative_response[:100]}...)"