import pickle,os
import pytz
from datetime import datetime
from typing import Any, List
from copy import deepcopy
from langchain.schema import Document

from langchain.retrievers import TimeWeightedVectorStoreRetriever

import faiss

from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings


class TimeWeightedVectorStoreRetrieverWithPersistence(TimeWeightedVectorStoreRetriever):
    persistent_path: str

    @classmethod
    def create_time_weighted_retriever(cls, persistent_dir: str = "memory", id: str = "default", k=1):
        """Create a new vector store retriever unique to the agent."""
        # Define your embedding model
        embeddings_model = OpenAIEmbeddings()
        # Initialize the vectorstore as empty
        if not os.path.exists(persistent_dir):
            os.makedirs(persistent_dir)
        
        persistent_path = os.path.join(persistent_dir, id)

        if os.path.exists(persistent_path):
            vectorstore = FAISS.load_local(persistent_path, embeddings_model)
        else:
            embedding_size = 1536
            index = faiss.IndexFlatL2(embedding_size)
            vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        retriever = TimeWeightedVectorStoreRetrieverWithPersistence(vectorstore=vectorstore, k=k, persistent_path=persistent_path, search_kwargs={"k": k, "score_threshold": 0.6}) 
        return retriever

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Return documents that are relevant to the query."""
        current_time = datetime.now(pytz.timezone('Asia/Tokyo'))
        docs_and_scores = {
            doc.metadata["buffer_idx"]: (doc, self.default_salience)
            for doc in self.memory_stream[-self.k :]
        }
        # If a doc is considered salient, update the salience score
        docs_and_scores.update(self.get_salient_docs(query))
        rescored_docs = [
            (doc, self._get_combined_score(doc, relevance, current_time))
            for doc, relevance in docs_and_scores.values()
        ]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        result = []
        # Ensure frequently accessed memories aren't forgotten
        current_time = datetime.now(pytz.timezone('Asia/Tokyo'))
        for doc, _ in rescored_docs[: self.k]:
            # TODO: Update vector store doc once `update` method is exposed.
            buffered_doc = self.memory_stream[doc.metadata["buffer_idx"]]
            buffered_doc.metadata["last_accessed_at"] = current_time
            result.append(buffered_doc)
        return result

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        current_time = kwargs.get("current_time", datetime.now(pytz.timezone('Asia/Tokyo')))
        # Avoid mutating input documents
        dup_docs = [deepcopy(d) for d in documents]
        for i, doc in enumerate(dup_docs):
            if "last_accessed_at" not in doc.metadata:
                doc.metadata["last_accessed_at"] = current_time
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = current_time
            doc.metadata["buffer_idx"] = len(self.memory_stream) + i
        self.memory_stream.extend(dup_docs)
        return self.vectorstore.add_documents(dup_docs, **kwargs)


    def save_memory_stream(self) -> None:
        """Save the memory stream to disk."""
        if not os.path.exists(self.persistent_path):
            os.makedirs(self.persistent_path)

        memory_stream_path = os.path.join(self.persistent_path, "memory_stream.pkl")

        with open(memory_stream_path, 'wb') as file:
            pickle.dump(self.memory_stream, file)
        self.vectorstore.save_local(self.persistent_path) 

    def load_memory_stream(self) -> None:
        """Load the memory stream."""

        memory_stream_path = os.path.join(self.persistent_path, "memory_stream.pkl")
        if os.path.exists(memory_stream_path):
            with open(memory_stream_path, 'rb') as file:
                self.memory_stream = pickle.load(file)
