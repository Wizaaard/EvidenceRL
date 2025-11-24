"""EvidenceRL: Retrieval-augmented generation with evidence-based rewards."""

from .baseline import (
    PatientCase,
    PromptPrediction,
    PromptingPredictor,
    load_patient_cases,
    patient_cases_to_rag_queries,
    summarise_predictions,
)
from .documents import CARDIAC_ICD_PREFIXES, Document, RetrievedDocument, load_cardiac_icd_documents
from .ingestion import (
    KnowledgeChunk,
    chunk_guideline_text,
    export_documents_jsonl,
    load_documents_from_jsonl,
    load_pdf_knowledge_documents,
)
from .evaluation import AnswerJudge, LLMAnswerJudge
from .generation import HuggingFaceGenerator
from .retrieval import DocumentStore, SemanticRetriever
from .pipeline import RagRlPipeline, RagRlResult

__all__ = [
    "CARDIAC_ICD_PREFIXES",
    "Document",
    "RetrievedDocument",
    "load_cardiac_icd_documents",
    "KnowledgeChunk",
    "chunk_guideline_text",
    "export_documents_jsonl",
    "load_documents_from_jsonl",
    "load_pdf_knowledge_documents",
    "PatientCase",
    "PromptPrediction",
    "PromptingPredictor",
    "load_patient_cases",
    "patient_cases_to_rag_queries",
    "summarise_predictions",
    "AnswerJudge",
    "LLMAnswerJudge",
    "HuggingFaceGenerator",
    "DocumentStore",
    "SemanticRetriever",
    "RagRlPipeline",
    "RagRlResult",
]
