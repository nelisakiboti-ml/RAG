import os
import time
import requests
from tqdm import tqdm

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp


class RAGPipelineApp:
    def __init__(self, model_path="llama-2-7b-chat.Q4_K_M.gguf", docs_dir="docs"):
        self.model_path = model_path
        self.docs_dir = docs_dir
        self.template = """
        Answer the question based on the following context:

        {context}

        Question: {question}
        Answer:
        """
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )

        self._download_model_if_needed()
        self._prepare_documents()
        self._build_vectorstore()
        self._initialize_llm()
        self._setup_pipeline()

    def _download_model_if_needed(self):
        if not os.path.exists(self.model_path):
            print(f"Downloading {self.model_path}...")
            model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
            response = requests.get(model_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(self.model_path, 'wb') as f:
                for data in tqdm(response.iter_content(chunk_size=1024), total=total_size // 1024):
                    f.write(data)
            print("Download complete!")

    def _prepare_documents(self):
        os.makedirs(self.docs_dir, exist_ok=True)
        sample_file = os.path.join(self.docs_dir, "sample.txt")
        if not os.path.exists(sample_file):
            with open(sample_file, "w") as f:
                f.write("""
                Retrieval-Augmented Generation (RAG) is a technique that combines retrieval-based and generation-based approaches
                for natural language processing tasks. It involves retrieving relevant information from a knowledge base and then 
                using that information to generate more accurate and informed responses.

                RAG models first retrieve documents that are relevant to a given query, then use these documents as additional context
                for language generation. This approach helps to ground the model's responses in factual information and reduces hallucinations.

                The llama.cpp library is a C/C++ implementation of Meta's LLaMA model, optimized for CPU usage. It allows running LLaMA models
                on consumer hardware without requiring high-end GPUs.

                LocalAI is a framework that enables running AI models locally without relying on cloud services. It provides APIs compatible
                with OpenAI's interfaces, allowing developers to use their own models with the same code they would use for OpenAI services.
                """)

        self.documents = []
        for file in os.listdir(self.docs_dir):
            file_path = os.path.join(self.docs_dir, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                self.documents.extend(loader.load())
            elif file.endswith(".txt"):
                loader = TextLoader(file_path)
                self.documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.chunks = splitter.split_documents(self.documents)

    def _build_vectorstore(self):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

    def _initialize_llm(self):
        self.llm = LlamaCpp(
            model_path=self.model_path,
            temperature=0.7,
            max_tokens=2000,
            n_ctx=4096,
            verbose=False
        )

    def _setup_pipeline(self):
        self.rag_pipeline = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def ask_question(self, question):
        start_time = time.time()
        result = self.rag_pipeline({"query": question})
        end_time = time.time()

        print(f"Question: {question}")
        print(f"Answer: {result['result']}")
        print(f"Time taken: {end_time - start_time:.2f} seconds\n")

        print("Source documents:")
        for i, doc in enumerate(result["source_documents"]):
            print(f"Document {i + 1}:")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content: {doc.page_content[:150]}...\n")


def main():
    app = RAGPipelineApp()
    app.ask_question("What is RAG and how does it work?")
    # app.ask_question("What is llama.cpp?")
    # app.ask_question("How does LocalAI relate to cloud AI services?")


if __name__ == "__main__":
    main()
