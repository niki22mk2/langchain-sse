import os, json
from typing import List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

from langchain.schema import messages_from_dict, messages_to_dict

from callback import CustomAsyncIteratorCallbackHandler
from memory import ConversationTokenBufferVectorMemory
from template import PROMPT

from retriever import TimeWeightedVectorStoreRetrieverWithPersistence

from agent import zeroshot

class ConversationManager:
    def __init__(self, conversation_id: str, input_variables: Optional[List[str]] = None, temperature: Optional[float] = 0.7, timeout: Optional[float] = 60.0, model: Optional[str] = "gpt-3.5-turbo", max_tokens: Optional[int] = None, ai_name: Optional[str] = "AI", human_name: Optional[str] = "Human"):
        self.conversation_id = conversation_id
        self.input_variables = input_variables
        self.input_variables.append("system")
        chat = ChatOpenAI(
            temperature=temperature, 
            streaming=True, 
            model_name=model, 
            request_timeout=timeout,
            max_tokens=max_tokens
        )
        self.retriever=TimeWeightedVectorStoreRetrieverWithPersistence.create_time_weighted_retriever(id=self.conversation_id)
        self.chain = ConversationChain(
            llm=chat,
            memory=ConversationTokenBufferVectorMemory(
                return_messages=True, 
                retriever=self.retriever,
                input_variables=self.input_variables,
                max_token_limit=600,
                ai_prefix=ai_name,
                human_prefix=human_name
            ),
            prompt=PROMPT
        )
        self.load_conversation()

    def save_conversation(self) -> None:
        memory_dict = messages_to_dict(self.chain.memory.chat_memory.messages)
        
        if not os.path.exists("memory"):
            os.makedirs("memory")

        with open(f"memory/{self.conversation_id}.json", "w") as f:
            json.dump(memory_dict, f, ensure_ascii=False, indent=4)

        self.retriever.save_memory_stream()

    def load_conversation(self) -> None:
        if os.path.exists(f"memory/{self.conversation_id}.json"):
            with open(f"memory/{self.conversation_id}.json", "r") as f:
                self.chain.memory.chat_memory.messages = messages_from_dict(json.load(f))

        self.retriever.load_memory_stream()

    async def generate_message(self, stream_handler: CustomAsyncIteratorCallbackHandler,system: str, input: str, **kwargs) -> str:
        
        if set(kwargs.keys()) != (set(self.input_variables) - {"system"}):
            raise ValueError("Some required input variables are missing or extraneous variables are provided")

        return await self.chain.apredict(input=input, system=system, callbacks=[stream_handler], **kwargs)

    def zeroshot_agent(self, user_message):
        from langchain.schema import get_buffer_string

        recent_history = get_buffer_string(self.chain.memory.chat_memory.messages[-3:])
        try:
            res = zeroshot(user_message, history=recent_history)
        except:
            res = None

        return res