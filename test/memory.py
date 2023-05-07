import tiktoken
from typing import Any, Dict, List, Optional, Union
from pydantic import Field

from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import BaseMessage, get_buffer_string, Document
from langchain.memory.utils import get_prompt_input_key

from retriever import TimeWeightedVectorStoreRetrieverWithPersistence

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

    def _get_prompt_input_key(self, inputs: Dict[str, Any]) -> str:
        """Get the input key for the prompt."""
        if self.input_key is None:
            return get_prompt_input_key(inputs, self.memory_variables)
        return self.input_key

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_key = self._get_prompt_input_key(inputs)
        query = inputs[input_key]
        try:
          docs = self.retriever.get_relevant_documents(query)
          result: Union[List[Document], str]
          result = "\n".join([doc.page_content for doc in docs])
        except:
          result = ""
        print("relevant_doc\n" + result)

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
        return {self.memory_key: final_buffer, self.relevant_key: result}

    def _form_documents(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> List[Document]:
        """Format context from this conversation to buffer."""
        excluded_keys = self.memory_variables
        # excluded_keys = [self.memory_key, self.relevant_key] + self.input_variables

        # Filter inputs by excluding the specified keys
        filtered_inputs = {k: v for k, v in inputs.items() if k not in excluded_keys}

        # Add human_prefix and ai_prefix to the respective keys
        input_texts = [f"{self.human_prefix if k == 'input' else k}: {v}" for k, v in filtered_inputs.items()]
        output_texts = [f"{self.ai_prefix if k == 'response' else k}: {v}" for k, v in outputs.items()]

        texts = input_texts + output_texts
        page_content = "\n".join(texts)
        return [Document(page_content=page_content)]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer. Pruned."""
        super().save_context(inputs, outputs)
        # Prune buffer if it exceeds max token limit
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        buffer = self.chat_memory.messages
        curr_buffer_length = sum([len(encoding.encode(get_buffer_string([m]))) for m in buffer])
        print("memory token length: ", curr_buffer_length)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = sum([len(encoding.encode(get_buffer_string([m]))) for m in buffer])

        print(inputs.get('information'))
        print(inputs.get('relevant'))

        documents = self._form_documents(inputs, outputs)
        self.retriever.add_documents(documents)