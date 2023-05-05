import asyncio
import json
from typing import Any, Dict, List, Union, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import messages_from_dict

import dotenv
dotenv.load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    model: Optional[str] = "gpt-3.5-turbo"
    timeout: Optional[float] = 60.0

class CustomAsyncIteratorCallbackHandler(AsyncIteratorCallbackHandler):

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.queue.put_nowait(json.dumps({
            "choices": [{"delta": {"content": token}}],
        }))

def convert_messages(messages: List[Dict[str, str]]) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
    # Converts a list of chat messages from a chatcompletion format to a LangChain format.
    return [
        {
            "type": "human" if item["role"] == "user" else "ai",
            "data": {
                "content": item["content"],
                "additional_kwargs": {},
            },
        }
        for item in messages
    ]

async def start_llm(stream_handler: CustomAsyncIteratorCallbackHandler, request: ChatRequest) -> None:
    chat = ChatOpenAI(
        temperature=request.temperature, 
        streaming=True, 
        model=request.model, 
        request_timeout=request.timeout
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(request.messages[0]["content"]),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    conversation = ConversationChain(
        llm=chat,
        memory=ConversationBufferMemory(return_messages=True),
        prompt=prompt
    )

    conversation.memory.chat_memory.messages = messages_from_dict(convert_messages(request.messages[1:-1]))

    await conversation.apredict(input=request.messages[-1]["content"], callbacks=[stream_handler])

@app.post("/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    stream_handler = CustomAsyncIteratorCallbackHandler()

    asyncio.create_task(start_llm(stream_handler, request))

    async def event_generator(acallback: CustomAsyncIteratorCallbackHandler):
        ait = acallback.aiter()

        async for token in ait:
            yield token

        yield "[DONE]"

    return EventSourceResponse(event_generator(stream_handler))
