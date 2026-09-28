import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.storage import document_manifest

load_dotenv()

# Set up folder paths
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
PDF_DIR = os.path.join(BASE_DIR, "data", "raw_pdfs")
DB_DIR = os.path.join(BASE_DIR, "data", "chromadb")

_embeddings = None
_vector_db = None


def _get_vector_db():
    """Lazily initialized singleton — mirrors the same pattern used in
    market_tools.py, so embeddings aren't reloaded from Hugging Face Hub on
    every ingestion run."""
    global _embeddings, _vector_db
    if _vector_db is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vector_db = Chroma(persist_directory=DB_DIR,
                            embedding_function=_embeddings)
    return _vector_db


def list_pending_pdfs() -> list[str]:
    """PDFs present in data/raw_pdfs/ that have NOT been ingested yet."""
    document_manifest.init_db()
    if not os.path.isdir(PDF_DIR):
        return []
    all_pdfs = sorted(f for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))
    return [f for f in all_pdfs if not document_manifest.is_ingested(f)]


def remove_document(filename: str) -> None:
    """Deletes a document's chunks from ChromaDB and forgets it in the manifest,
    so it can be re-ingested from scratch if re-uploaded."""
    vector_db = _get_vector_db()
    try:
        vector_db._collection.delete(
            where={"source": os.path.join(PDF_DIR, filename)})
    except Exception:
        pass
    document_manifest.forget_document(filename)


def process_and_store_pdfs(filenames: list[str] | None = None):
    """
    Generator that processes PDFs one at a time and yields progress events, so
    a caller (e.g. the Streamlit UI) can show live per-stage status instead of
    a single opaque spinner.

    filenames: specific files to (re)process. If None, processes every PDF in
    data/raw_pdfs/ that isn't already in the ingestion manifest — repeated
    clicks on "Process and Index" therefore no longer duplicate chunks for
    files already indexed.

    Yields dicts: {"stage": ..., "filename": ..., "current": i, "total": n, ...}
    Stages: "loading", "chunking", "embedding", "file_done", "all_done", "error"
    """
    document_manifest.init_db()

    if filenames is None:
        pdf_files = list_pending_pdfs()
    else:
        pdf_files = filenames

    if not pdf_files:
        yield {"stage": "nothing_to_do"}
        return

    vector_db = _get_vector_db()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    total = len(pdf_files)
    for i, pdf_file in enumerate(pdf_files, start=1):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        try:
            yield {"stage": "loading", "filename": pdf_file, "current": i, "total": total}
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            yield {"stage": "chunking", "filename": pdf_file, "current": i, "total": total}
            chunks = text_splitter.split_documents(documents)

            yield {
                "stage": "embedding", "filename": pdf_file, "current": i, "total": total,
                "chunk_count": len(chunks),
            }
            vector_db.add_documents(chunks)

            document_manifest.mark_ingested(pdf_file, len(chunks))
            yield {
                "stage": "file_done", "filename": pdf_file, "current": i, "total": total,
                "chunk_count": len(chunks),
            }
        except Exception as e:
            yield {"stage": "error", "filename": pdf_file, "current": i, "total": total, "error": str(e)}

    yield {"stage": "all_done", "total": total}


if __name__ == "__main__":
    for event in process_and_store_pdfs():
        print(event)
