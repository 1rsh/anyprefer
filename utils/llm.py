import instructor
from openai import AsyncOpenAI
from abc import ABC, abstractmethod
from typing import List, Literal, Union
from pydantic import BaseModel

from utils.logger import logger
from utils.constants import PROVIDER_INFORMATION
from utils.tools.base import BaseTool

class BaseLLMService(ABC):
    @abstractmethod
    async def call_llm(self, model: str, messages: List[dict]):
        pass

    @abstractmethod
    async def call_llm_structured(self, model: str, messages: List[dict], response_format: BaseModel):
        pass

    @abstractmethod
    async def call_llm_tools(self, model: str, messages: List[dict], tools: List[dict], tool_choice: Union[Literal['auto', 'none'], dict] = 'auto'):
        pass

class LLMService(BaseLLMService):
    def __init__(self, name: str):
        self.name = name
        api_key, base_url = PROVIDER_INFORMATION[name]["API"]
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=2
        )

    def _get_model_id(self, model: str):
        return model, PROVIDER_INFORMATION[self.name]["MODEL_ID"][model]

    async def call_llm(self, model: str, messages: List[dict]):
        """Call the LLM with the given model and messages."""
        generic_model_name, model = self._get_model_id(model)
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return response.choices[0].message.content if response.choices else None
        except Exception as e:
            logger.error(f"{self.name} LLM service failed to call model {generic_model_name}: {e}")
            return None

    async def call_llm_structured(self, model: str, messages: List[dict], response_format: BaseModel):
        """Call the LLM with the given model and messages."""
        generic_model_name, model = self._get_model_id(model)
        try:
            structured_client = instructor.from_openai(self.client, mode=instructor.Mode.JSON)
            response = await structured_client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=response_format,
            )
            return response
        except Exception as e:
            logger.error(f"{self.name} LLM service failed to call model {generic_model_name}: {e}")
            return None
    
    async def call_llm_tools(self, model: str, messages: List[dict], tools: dict[str, BaseTool], tool_choice: Union[Literal['auto', 'none'], dict] = 'auto'):
        generic_model_name, model = self._get_model_id(model)
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[tool().openai_dict for tool in tools.values()],
                tool_choice=tool_choice
            )

            if response.choices and len(response.choices) > 0 and hasattr(response.choices[0].message, 'tool_calls'):
                tool_response = response.choices[0].message.tool_calls
                chosen_tool = tools.get(tool_response[0].function.name)()
                context = await chosen_tool.run(**tool_response[0].function.arguments)
                return context
            else:
                return None
        except Exception as e:
            logger.error(f"{self.name} LLM service failed to call model {generic_model_name}: {e}")
            return None
