from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains.retrieval_qa import RetrievalQA
from langchain_openai import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever

load_dotenv()

embeddings = OpenAIEmbeddings()

db = Chroma(persist_directory="db/facts", embedding_function=embeddings)

# retriever = db.as_retriever()
retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db,
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=1000,
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

result = chain.run("What is an interesting fact about the Tour de France?")
print(result)


def search(query, k=3):
    results = db.similarity_search(query=query, k=k)
    return results
