import asyncio
from typing import List

from prompts import (
    TARGET_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT,
    REWARD_SYSTEM_PROMPT,
    REWARD_USER_PROMPT,
    PROMPT_SYSTEM_PROMPT,
    PROMPT_USER_PROMPT,
)

from utils.llm import LLMService
from utils.logger import logger
from utils.constants import OPENAI, GPT_4O_MINI
from utils.response_models import JudgeResponse, RewardResponse, PromptOptimizationResponse


class AnyPrefer:
    def __init__(self):
        self.llm_services = {
            "target": LLMService(OPENAI),
            "judge": LLMService(OPENAI),
            "reward": LLMService(OPENAI),
            "prompt": LLMService(OPENAI),
        }
        
        self.models = {
            "target": GPT_4O_MINI,
            "judge": GPT_4O_MINI,
            "reward": GPT_4O_MINI,
            "prompt": GPT_4O_MINI,
        }

        self.target_system_prompt = TARGET_SYSTEM_PROMPT
        self.judge_system_prompt = JUDGE_SYSTEM_PROMPT
        self.reward_threshold = 8

    async def generate_candidate_responses(self, prompt: str, num_responses: int = 5) -> List[str]:
        messages = [
            {"role": "system", "content": self.target_system_prompt},
            {"role": "user", "content": prompt}
        ]
        model = self.models["target"]
        llm_service = self.llm_services["target"]

        tasks = []

        for _ in range(num_responses):
            tasks.append(llm_service.call_llm(model=model, messages=messages))
        
        responses = await asyncio.gather(*tasks)
        return [response for response in responses if response is not None]
    
    async def judge_responses(self, prompt: str, responses: List[str]) -> List[JudgeResponse]:
        model = self.models["judge"]
        llm_service = self.llm_services["judge"]
        
        judge_results = []
        for response in responses:
            messages = [
                {"role": "system", "content": self.judge_system_prompt},
                {"role": "user", "content": JUDGE_USER_PROMPT.format(prompt=prompt, response=response)}
            ]
            judge_results.append(await llm_service.call_llm_structured(model=model, messages=messages, response_format=JudgeResponse))
        
        return [result or JudgeResponse(reasoning="", score=1) for result in judge_results]
    
    async def generate_reward_analysis(self, prompt: str, positive_response: str, negative_response: str) -> RewardResponse:
        model = self.models["reward"]
        llm_service = self.llm_services["reward"]
        
        messages = [
            {"role": "system", "content": REWARD_SYSTEM_PROMPT},
            {"role": "user", "content": REWARD_USER_PROMPT.format(prompt=prompt, positive_response=positive_response, negative_response=negative_response)}
        ]
        
        reward_response = await llm_service.call_llm_structured(model=model, messages=messages, response_format=RewardResponse)
        return reward_response or RewardResponse(analysis="", score=0)
    
    async def optimize_prompts(self):
        model = self.models["prompt"]
        llm_service = self.llm_services["prompt"]

        messages = [
            {"role": "system", "content": PROMPT_SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_USER_PROMPT.format(target_system_prompt=self.target_system_prompt, judge_system_prompt=self.judge_system_prompt)}
        ]

        prompt_response = await llm_service.call_llm_structured(model=model, messages=messages, response_format=PromptOptimizationResponse)
        if prompt_response:
            self.target_system_prompt = prompt_response.optimized_target_system_prompt
            self.judge_system_prompt = prompt_response.optimized_judge_system_prompt
        
        return
    
    async def run_one(self, prompt: str, num_iterations: int = 3):
        for i in range(num_iterations):
            logger.info(f"Iteration {i+1}/{num_iterations}")

            # Step 1: Generate candidate responses
            candidate_responses = await self.generate_candidate_responses(prompt, num_responses=5)

            if not candidate_responses:
                logger.warning("No candidate responses generated.")
                continue

            # Step 2: Judge the candidate responses
            judged_responses = await self.judge_responses(prompt, candidate_responses)

            # Step 3: Select the best and worst responses
            candidate_responses = sorted(zip(candidate_responses, judged_responses), key=lambda x: x[1].score, reverse=True)

            logger.debug(f"Judged Responses: {candidate_responses}")

            positive_response, _ = candidate_responses[0]
            negative_response, _ = candidate_responses[-1]

            logger.debug(f"Best Response: {positive_response} | Score: {candidate_responses[0][1].score}")
            logger.debug(f"Worst Response: {negative_response} | Score: {candidate_responses[-1][1].score}")

            # Step 4: Generate reward analysis
            reward_analysis = await self.generate_reward_analysis(prompt, positive_response, negative_response)
            logger.info(f"Reward Analysis: {reward_analysis.analysis} | Score: {reward_analysis.score}")

            if reward_analysis.score >= self.reward_threshold:
                self.target_system_prompt = TARGET_SYSTEM_PROMPT
                self.judge_system_prompt = JUDGE_SYSTEM_PROMPT
                return (prompt, positive_response, negative_response)
            else:
                logger.debug("Reward score below threshold, optimizing prompts...")
                await self.optimize_prompts()
                continue
        
        self.target_system_prompt = TARGET_SYSTEM_PROMPT
        self.judge_system_prompt = JUDGE_SYSTEM_PROMPT
        logger.critical("Max iterations reached without satisfactory response, consider reducing the reward threshold.")
        

if __name__ == "__main__":
    anyprefer = AnyPrefer()
    test_prompt = "What is the best way to deal with stress amongst students?"
    
    asyncio.run(anyprefer.run_one(test_prompt, num_iterations=3))