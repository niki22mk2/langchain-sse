
import json
from typing import Any

from langchain.callbacks import AsyncIteratorCallbackHandler

class CustomAsyncIteratorCallbackHandler(AsyncIteratorCallbackHandler):
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.queue.put_nowait(json.dumps({
            "choices": [{"delta": {"content": token}}],
        }))