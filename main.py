from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.vectorstores.chroma import Chroma

load_dotenv()


def main():
    # Define the embeddings, LLM, text splitter, and loader
    embeddings = OpenAIEmbeddings()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=1000,
    )

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=200, chunk_overlap=0
    )
    loader = TextLoader("data/facts.txt")

    docs = loader.load_and_split(text_splitter=text_splitter)

    # Chroma
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="facts",
        persist_directory="db/facts",
    )

    # for doc in docs:
    #     print(doc.page_content)
    #     print("\n")
    results = db.similarity_search(
        query="What is an interesting fact about the Tour de France?",
        k=3,
    )

    for result in results:
        print("\n")
        print(result.metadata)
        print(result.page_content)
        # print(result[0].page_content)
        print("\n---\n")


if __name__ == "__main__":
    main()
