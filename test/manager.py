import os, json
from typing import List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

from langchain.schema import messages_from_dict, messages_to_dict

from callback import CustomAsyncIteratorCallbackHandler
from memory import ConversationTokenBufferMemory, ConversationTokenBufferVectorMemory
from template import PROMPT



class ConversationManager:
    def __init__(self, conversation_id: str, input_variables: Optional[List[str]] = None, temperature: Optional[float] = 0.7, timeout: Optional[float] = 60.0, model: Optional[str] = "gpt-3.5-turbo", max_tokens: Optional[int] = 400):
        self.conversation_id = conversation_id
        self.input_variables = input_variables
        self.input_variables.append("system")
        chat = ChatOpenAI(
            temperature=temperature, 
            streaming=True, 
            model=model, 
            request_timeout=timeout,
            max_tokens=max_tokens
        )
        # self.chain = ConversationChain(
        #     llm=chat,
        #     memory=ConversationTokenBufferMemory(return_messages=True, input_variables=self.input_variables),
        #     prompt=PROMPT
        # )

        self.chain = ConversationChain(
            llm=chat,
            memory=ConversationTokenBufferVectorMemory(return_messages=True, retriever=vectorstore.as_retriever(),input_variables=self.input_variables),
            prompt=PROMPT
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

    async def generate_message(self, stream_handler: CustomAsyncIteratorCallbackHandler,system: str, input: str, **kwargs) -> str:
        
        if set(kwargs.keys()) != (set(self.input_variables) - {"system"}):
            raise ValueError("Some required input variables are missing or extraneous variables are provided")

        print(self.chain.memory.chat_memory.messages)
        return await self.chain.apredict(input=input, system=system, callbacks=[stream_handler], **kwargs)
