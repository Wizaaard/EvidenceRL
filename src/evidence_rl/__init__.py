"""EvidenceRL: Retrieval-augmented generation with evidence-based rewards."""

from .baseline import (
    PatientCase,
    PromptPrediction,
    PromptingPredictor,
    RAGPredictor,
    load_patient_cases,
    summarise_predictions,
)
from .documents import CARDIAC_ICD_PREFIXES, Document, RetrievedDocument, load_cardiac_icd_documents
from .ingestion import (
    KnowledgeChunk,
    chunk_guideline_text,
    export_documents_jsonl,
    load_documents_from_hf_dataset,
    load_documents_from_jsonl,
    load_pdf_knowledge_documents,
)
from .evaluation import LLMAnswerJudge
from .generation import HuggingFaceGenerator
from .retrieval import DocumentStore, SemanticRetriever, HuggingFaceEmbedder

# Optional: FAISS-based fast retrieval (requires faiss-cpu or faiss-gpu)
try:
    from .faiss_retrieval import FAISSDocumentStore, FAISSRetriever
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

# Optional: Alternative vector stores (require respective packages)
try:
    from .vector_stores import ChromaDBStore, QdrantStore, PineconeStore
    _HAS_VECTOR_STORES = True
except ImportError:
    _HAS_VECTOR_STORES = False

from .pipeline import RagRlPipeline, RagRlResult
from .evidence_pipeline import (
    DiagnosisWithReasoning,
    StructuredDiagnosisOutput,
    EvidenceRLResult,
    StructuredDiagnosisGenerator,
    EvidenceRLPipeline,
    summarize_results,
)

# Optional: Self-Consistency pipeline (requires vllm)
try:
    from .self_consistency_pipeline import (
        SelfConsistencyVLLMPipeline,
        SCMetadata,
        DiagnosisCluster,
        cluster_diagnoses_by_embedding,
        sc_result_to_dict,
    )
    _HAS_SC = True
except ImportError:
    _HAS_SC = False

__all__ = [
    # Documents
    "CARDIAC_ICD_PREFIXES",
    "Document",
    "RetrievedDocument",
    "load_cardiac_icd_documents",
    # Ingestion
    "KnowledgeChunk",
    "chunk_guideline_text",
    "export_documents_jsonl",
    "load_documents_from_hf_dataset",
    "load_documents_from_jsonl",
    "load_pdf_knowledge_documents",
    # Baseline
    "PatientCase",
    "PromptPrediction",
    "PromptingPredictor",
    "RAGPredictor",
    "load_patient_cases",
    "summarise_predictions",
    # Evaluation
    "LLMAnswerJudge",
    # Generation
    "HuggingFaceGenerator",
    # Retrieval
    "DocumentStore",
    "SemanticRetriever",
    "HuggingFaceEmbedder",
    # Fast retrieval (optional, requires faiss)
    *(['FAISSDocumentStore', 'FAISSRetriever'] if _HAS_FAISS else []),
    # Vector stores (optional, require respective packages)
    *(['ChromaDBStore', 'QdrantStore', 'PineconeStore'] if _HAS_VECTOR_STORES else []),
    # Original Pipeline
    "RagRlPipeline",
    "RagRlResult",
    # EvidenceRL Pipeline
    "DiagnosisWithReasoning",
    "StructuredDiagnosisOutput",
    "EvidenceRLResult",
    "StructuredDiagnosisGenerator",
    "EvidenceRLPipeline",
    "summarize_results",
    # Self-Consistency Pipeline (optional, requires vllm)
    *(['SelfConsistencyVLLMPipeline', 'SCMetadata', 'DiagnosisCluster',
       'cluster_diagnoses_by_embedding', 'sc_result_to_dict'] if _HAS_SC else []),
]
