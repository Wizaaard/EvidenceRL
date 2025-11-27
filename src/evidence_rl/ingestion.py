"""Utilities for turning external medical PDFs into retrievable documents.

The helpers in this module focus on the "bring your own knowledge" baseline
requested for the cardiac RAG workflow. They ingest PDF (or text) guidelines
from a directory, heuristically segment them into sections/subsections, chunk
the content for retrieval, and optionally persist the resulting corpus as
JSONL. The chunked :class:`~evidence_rl.documents.Document` objects can be
passed directly into :class:`~evidence_rl.retrieval.DocumentStore`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

from .documents import Document

_SECTION_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*\.?)\s+(.+)$")


def _extract_pdf_text(path: Path) -> str:
    """Extract raw text from a PDF or plaintext file.

    ``pypdf`` or ``PyPDF2`` is used when available; plaintext files can be used
    for lightweight testing and dry runs.
    """

    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")

    try:  # pragma: no cover - dependency is optional in CI
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:  # pragma: no cover - alternate fallback
            from PyPDF2 import PdfReader  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install pypdf or PyPDF2 to read PDF knowledge files.") from exc

    try:  # pragma: no cover - depends on availability of a PDF backend
        reader = PdfReader(str(path))
    except Exception as exc:  # pragma: no cover - runtime feedback
        raise RuntimeError(f"Failed to read PDF {path}: {exc}") from exc

    pages: List[str] = []
    for page in getattr(reader, "pages", []):
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def _strip_numbering(title: str) -> str:
    # Remove leading patterns like "1", "1.", "1.2", "1.2.3", "2.1." etc.
    return re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", title).strip()

def _iter_sections(text: str) -> Iterator[Tuple[str, str]]:
    """Yield ``(title, body)`` pairs for each detected section.

    A section heading is detected if a line is:
    * Numbered (e.g., ``1 Introduction`` or ``2.3 Treatment``), or
    * Uppercase and reasonably short (heuristic for clinical guideline titles).
    """

    lines = text.splitlines()
    current_title = "preamble"
    buffer: List[str] = []

    sections: List[Tuple[str, str]] = []

    for line in lines:
        heading: str | None = None
        
        # numbered headings
        match = _SECTION_HEADING_RE.match(line)
        if match:
            heading = match.group(0).strip()
        
        # ALL CAPS headings
        elif line.strip().isupper() and 3 <= len(line.strip()) <= 120:
            heading = line.strip()

        # If heading found and we have collected buffer from previous section
        if heading and buffer:
            # prepend title to body
            clean_title = _strip_numbering(current_title)
            section_body = f"{clean_title}\n" + "\n".join(buffer).strip()
            sections.append((current_title, section_body))
            buffer = []
            current_title = heading
            continue

        buffer.append(line)

    # last section
    if buffer:
        clean_title = _strip_numbering(current_title)
        section_body = f"{clean_title}\n" + "\n".join(buffer).strip()
        sections.append((current_title, section_body))

    if not sections and text.strip():
        yield "full_document", text.strip()
        return

    for title, body in sections:
        if body:
            yield title or "preamble", body


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Chunk text within a single section.

    - If len(tokens) <= chunk_size: return [text] as a single chunk.
    - Else: sliding window with overlap, constrained to this text only.
    """
    tokens = text.split()
    n_tokens = len(tokens)
    if n_tokens == 0:
        return []

    # If the whole section fits into one chunk, don't bother with overlap
    if n_tokens <= chunk_size:
        return [" ".join(tokens)]

    chunks: List[str] = []
    start = 0
    while start < n_tokens:
        end = min(n_tokens, start + chunk_size)
        chunks.append(" ".join(tokens[start:end]))
        if end == n_tokens:
            break
        start = end - overlap  # overlap within this section only
    return chunks


@dataclass(frozen=True)
class KnowledgeChunk:
    doc_id: str
    text: str
    metadata: Dict[str, object]

    def to_document(self) -> Document:
        return Document(doc_id=self.doc_id, text=self.text, metadata=self.metadata)


def chunk_guideline_text(
    text: str,
    *,
    source_id: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[KnowledgeChunk]:
    """Chunk a guideline's text into section-aware pieces for retrieval.

    - Each section from `_iter_sections` is chunked independently.
    - No chunk ever spans multiple sections.
    """

    chunks: List[KnowledgeChunk] = []
    section_index = 0
    for section_title, section_body in _iter_sections(text):
        section_chunks = _chunk_text(section_body, chunk_size=chunk_size, overlap=overlap)
        if not section_chunks:
            continue
        for idx, chunk in enumerate(section_chunks):
            doc_id = f"{source_id}::section-{section_index}::chunk-{idx}"
            metadata = {
                "source_id": source_id,
                "section_title": section_title,
                "section_index": section_index,
                "chunk_index": idx,
            }
            chunks.append(KnowledgeChunk(doc_id=doc_id, text=chunk, metadata=metadata))
        section_index += 1

    return chunks


def load_pdf_knowledge_documents(
    knowledge_dir: str | Path,
    *,
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[Document]:
    """Load and chunk all PDFs (or text files) under ``knowledge_dir``.

    The resulting :class:`Document` list is ready for dense retrieval and can
    optionally be persisted using :func:`export_documents_jsonl`.
    """

    base = Path(knowledge_dir)
    if not base.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {base}")

    files = sorted([p for p in base.glob("**/*") if p.suffix.lower() in {".txt"}])
    if not files:
        raise ValueError(f"No PDF or text files found under {base}")

    documents: List[Document] = []
    for path in files:
        text = _extract_pdf_text(path)
        source_id = path.stem
        chunks = chunk_guideline_text(text, source_id=source_id, chunk_size=chunk_size, overlap=overlap)
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata.update({
                "source_path": str(path),
                "source_filename": path.name,
            })
            documents.append(Document(doc_id=chunk.doc_id, text=chunk.text, metadata=metadata))

    if not documents:
        raise ValueError(f"No retrievable chunks could be produced from {base}")

    return documents


def export_documents_jsonl(documents: Iterable[Document], output_path: str | Path) -> None:
    """Write a JSONL corpus with one row per document chunk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc in documents:
            payload = {"doc_id": doc.doc_id, "text": doc.text, "metadata": doc.metadata or {}}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_documents_from_jsonl(jsonl_path: str | Path) -> List[Document]:
    """Load :class:`Document` objects from a JSONL file produced by export."""

    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL corpus not found: {path}")

    documents: List[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            documents.append(
                Document(
                    doc_id=payload["doc_id"],
                    text=payload["text"],
                    metadata=payload.get("metadata") or {},
                )
            )

    if not documents:
        raise ValueError(f"No documents loaded from {path}")

    return documents


__all__ = [
    "KnowledgeChunk",
    "chunk_guideline_text",
    "export_documents_jsonl",
    "load_documents_from_jsonl",
    "load_pdf_knowledge_documents",
]
