import tiktoken
from typing import Any, Dict, List, Optional, Union
from pydantic import Field

from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import BaseMessage, get_buffer_string, Document
from langchain.memory.utils import get_prompt_input_key
from langchain.schema import BaseChatMessageHistory, BaseMemory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from retriever import TimeWeightedVectorStoreRetrieverWithPersistence
from tools.utils import get_date
from template import MEMORY_PROMPT

import dotenv
dotenv.load_dotenv()


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
    llm: ChatOpenAI = ChatOpenAI(temperature=0)
    chat_memory_summarize: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)

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
        
        input_str, output_str = self._get_input_output(inputs, outputs)
        time = get_date(raw=True).strftime('%Y/%m/%d %H:%M')
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)

        self.chat_memory_summarize.add_user_message(f'{time} {input_str}')
        self.chat_memory_summarize.add_ai_message(f'{time} {output_str}')

        # Prune buffer if it exceeds max token limit
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        buffer = self.chat_memory.messages
        buffer_summarize = self.chat_memory_summarize.messages
        curr_buffer_length = sum([len(encoding.encode(get_buffer_string([m]))) for m in buffer])
        print("memory token length: ", curr_buffer_length)
        print("\n".join([get_buffer_string([m], human_prefix=self.human_prefix, ai_prefix=self.ai_prefix) for m in buffer]))
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            half_length = len(buffer) // 2
            pruned_memory.extend(buffer[:half_length])
            buffer = buffer[half_length:]
            buffer_summarize = buffer_summarize[half_length:]

            llm_chain = LLMChain(llm=self.llm, prompt=MEMORY_PROMPT, verbose=True)
            memories_summarize = llm_chain.predict(name=self.ai_prefix, history="\n".join([get_buffer_string([m], human_prefix=self.human_prefix, ai_prefix=self.ai_prefix) for m in buffer_summarize]))
            print(memories_summarize)
            for memory in memories_summarize.split("\n"):
                self.retriever.add_documents([Document(page_content=memory)])


        print(inputs.get('information'))
        print(inputs.get('relevant'))

        # documents = self._form_documents(inputs, outputs)
        # self.retriever.add_documents(documents)