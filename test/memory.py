from typing import Any, Dict, List, Optional

from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import BaseMessage, get_buffer_string
from langchain.memory import VectorStoreRetrieverMemory

import tiktoken

class ConversationTokenBufferMemory(BaseChatMemory):
    """Buffer for storing conversation memory."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"
    max_token_limit: int = 2000
    input_variables: Optional[List[str]] = None

    @property
    def buffer(self) -> List[BaseMessage]:
        """String buffer of memory."""
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        if self.input_variables:
            return [self.memory_key, *self.input_variables]
        else:
            return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        buffer: Any = self.buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer. Pruned."""
        super().save_context(inputs, outputs)
        # Prune buffer if it exceeds max token limit
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        buffer = self.chat_memory.messages
        curr_buffer_length = sum([len(encoding.encode(get_buffer_string([m]))) for m in buffer])
        print(curr_buffer_length)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = sum([len(encoding.encode(get_buffer_string([m]))) for m in buffer])


from pydantic import Field

from langchain.memory.chat_memory import BaseMemory
from langchain.memory.utils import get_prompt_input_key
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from retriever import TimeWeightedVectorStoreRetrieverWithPersistence

class ConversationTokenBufferVectorMemory(BaseChatMemory):
    """Buffer for storing conversation memory."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"
    relevant_key: str = "relevant" 
    max_token_limit: int = 2000
    input_variables: Optional[List[str]] = None
    retriever: TimeWeightedVectorStoreRetrieverWithPersistence = Field(exclude=True)

    @property
    def buffer(self) -> List[BaseMessage]:
        """String buffer of memory."""
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        if self.input_variables:
            return [self.memory_key, self.relevant_key, *self.input_variables]
        else:
            return [self.memory_key, self.relevant_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        buffer: Any = self.buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer. Pruned."""
        super().save_context(inputs, outputs)
        # Prune buffer if it exceeds max token limit
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        buffer = self.chat_memory.messages
        curr_buffer_length = sum([len(encoding.encode(get_buffer_string([m]))) for m in buffer])
        print(curr_buffer_length)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = sum([len(encoding.encode(get_buffer_string([m]))) for m in buffer])
