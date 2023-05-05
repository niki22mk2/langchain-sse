import os, json
from typing import List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

from langchain.prompts import (
    ChatPromptTemplate
)

from langchain.schema import messages_from_dict, messages_to_dict

from callback import CustomAsyncIteratorCallbackHandler
from memory import ConversationTokenBufferMemory

class ConversationManager:
    def __init__(self, conversation_id: str, prompt: ChatPromptTemplate, input_variables: Optional[List[str]] = None, temperature: Optional[float] = 0.7, timeout: Optional[float] = 60.0, model: Optional[str] = "gpt-3.5-turbo"):
        self.conversation_id = conversation_id
        chat = ChatOpenAI(
            temperature=temperature, 
            streaming=True, 
            model=model, 
            request_timeout=timeout
        )
        self.chain = ConversationChain(
            llm=chat,
            memory=ConversationTokenBufferMemory(return_messages=True, input_variables=input_variables),
            prompt=prompt
        )
        self.load_conversation()

    def save_conversation(self) -> None:
        memory_dict = messages_to_dict(self.chain.memory.chat_memory.messages)
        
        if not os.path.exists("memory"):
            os.makedirs("memory")

        with open(f"memory/{self.conversation_id}.json", "w") as f:
            json.dump(memory_dict, f, ensure_ascii=False, indent=4)

    def load_conversation(self) -> None:
        if os.path.exists(f"memory/{self.conversation_id}.json"):
            with open(f"memory/{self.conversation_id}.json", "r") as f:
                self.chain.memory.chat_memory.messages = messages_from_dict(json.load(f))

    async def generate_message(self, stream_handler: CustomAsyncIteratorCallbackHandler, input: str, **kwargs) -> str:
        print(self.chain.memory.chat_memory.messages)
        return await self.chain.apredict(input=input, callbacks=[stream_handler], **kwargs)
