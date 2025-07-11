from dotenv import load_dotenv

from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever

load_dotenv()


class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def __init__(self, retriever: BaseRetriever, k: int = 3):
        self.retriever = retriever
        self.k = k

    def get_relevant_documents(self, query: str):
        # calculate emebeddings for the 'query' string
        emb = self.embeddings.embed_query(query)
        # take embeddings and feed them into that
        # max_marginal_relevance_search_by_vector
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb, lambda_mult=0.8
        )

    async def aget_relevant_documents(self, query: str):
        results = await self.retriever.aget_relevant_documents(query)
        return results[: self.k]
