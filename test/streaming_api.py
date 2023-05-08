import asyncio
from typing import Dict, List, Optional

import dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from functools import lru_cache

from callback import CustomAsyncIteratorCallbackHandler
from manager import ConversationManager

from tools.utils import get_date

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
    max_tokens: Optional[int] = None
    conversation_id: Optional[str] = "default"
    ai_name: Optional[str] = "AI",
    human_name: Optional[str] = "Human"


@lru_cache(maxsize=10)
def get_conversation_manager(
    conversation_id: str, temperature: Optional[float] = 0.7, timeout: Optional[float] = 60.0, model: Optional[str] = "gpt-3.5-turbo", max_tokens: Optional[int] = None, ai_name: Optional[str] = "AI", human_name: Optional[str] = "Human"
) -> ConversationManager:

    return ConversationManager(
        conversation_id=conversation_id,
        input_variables=["information", "date"],
        temperature=temperature,
        timeout=timeout,
        model=model,
        max_tokens=max_tokens,
        ai_name=ai_name,
        human_name=human_name,
    )


async def start_llm(stream_handler: CustomAsyncIteratorCallbackHandler, request: ChatRequest) -> None:
    system_message = request.messages[0]["content"]
    user_message = request.messages[-1]["content"]

    conversation_manager = get_conversation_manager(
        conversation_id=request.conversation_id,
        temperature=request.temperature,
        timeout=request.timeout,
        model=request.model,
        ai_name=request.ai_name,
        human_name=request.human_name,
    )

    information = conversation_manager.zeroshot_agent(user_message)

    print("use model:", request.model)

    await conversation_manager.generate_message(stream_handler=stream_handler, input=user_message, system=system_message, information=information, date=get_date())

    conversation_manager.save_conversation()


@app.on_event("startup")
async def startup() -> None:
    print("LangChain API is ready")


@app.post("/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    stream_handler = CustomAsyncIteratorCallbackHandler()

    asyncio.create_task(start_llm(stream_handler, request))

    async def event_generator(acallback: CustomAsyncIteratorCallbackHandler):
        ait = acallback.aiter()

        print("Starting stream")
        async for token in ait:
            yield token

        print("Stream finished")
        yield "[DONE]"

    return EventSourceResponse(event_generator(stream_handler))
