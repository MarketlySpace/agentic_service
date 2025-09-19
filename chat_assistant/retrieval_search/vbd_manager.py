import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from chat_assistant.global_config import (
    VECTOR_DB_DIR,
    EMBEDDER_CHUNK_SIZE,
    TIMEOUT,
    RETRY_REQUEST_TIME,
    NAME_EMBEDDER_MODEL,
    SPLITER_CHUNK_SIZE,
    SPLITER_OVERLAP,
    K_SAMPLES,
)


class VBDManager:

    def __init__(
            self,
            persist_directory: str = VECTOR_DB_DIR,
    ) -> None:
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(
            deployment=NAME_EMBEDDER_MODEL,
            chunk_size=EMBEDDER_CHUNK_SIZE,
            timeout=TIMEOUT,
            show_progress_bar=True,
            retry_min_seconds=RETRY_REQUEST_TIME,
        )
        if os.path.exists(persist_directory):
            self.db = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
        else:
            raise FileNotFoundError(
                f"Vector DB not found at '{persist_directory}'. "
                f"Run create_db(...) first to initialize."
            )

    def _load_documents(self, source_type, path_or_url):
        match source_type:
            case "pdf":
                loader = PyPDFLoader(path_or_url)
                docs = loader.load()
            case "gdoc":
                export_url = f"{path_or_url}/export?format=txt"
                loader = UnstructuredURLLoader(urls=[export_url])
                docs = loader.load()
            case _:
                raise ValueError(f"Unknown source_type: {source_type}")
        return docs

    def create_db(self, sources: list[tuple[str, str]]):
        all_docs = []
        for source_type, path_or_url in sources:
            raw_documents = self._load_documents(source_type, path_or_url)
            splitter = CharacterTextSplitter(
                chunk_size=SPLITER_CHUNK_SIZE,
                chunk_overlap=SPLITER_OVERLAP,
            )
            all_docs.extend(splitter.split_documents(raw_documents))
        self.db = Chroma.from_documents(
            all_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.db.persist()
        return self.db

    def add_documents(self, source_type, path_or_url):
        if not self.db:
            raise ValueError("DB is not initialized. Use create_db() first.")
        raw_documents = self._load_documents(source_type, path_or_url)
        splitter = CharacterTextSplitter(
            chunk_size=SPLITER_CHUNK_SIZE,
            chunk_overlap=SPLITER_OVERLAP,
        )
        docs = splitter.split_documents(raw_documents)
        self.db.add_documents(docs)
        self.db.persist()

    def delete_document(self, doc_id):
        if not self.db:
            raise ValueError("DB is not initialized.")
        self.db.delete(ids=[doc_id])
        self.db.persist()

    def update_document(self, doc_id, source_type, path_or_url):
        self.delete_document(doc_id)
        self.add_documents(source_type, path_or_url)

    def search(self, query, k=K_SAMPLES):
        if not self.db:
            raise ValueError("DB is not initialized.")
        return self.db.similarity_search(query, k=k)
