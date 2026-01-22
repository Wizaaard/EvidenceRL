# EvidenceRL: Reinforcing Evidence Consistency for Trustworthy Language Models

**Anonymous ICML 2026 Submission Draft (Internal)**

## Abstract
We present **EvidenceRL**, a reinforcement learning framework that increases evidence adherence in retrieval-augmented generation (RAG). The key idea is to run two retrieval passes: (1) **input-time retrieval** to obtain top-*$k$ evidence for the user query and (2) **output-time retrieval** to obtain top-*$k$ evidence for the model’s generated answer. We define an **evidence consistency score** based on the similarity between the two retrieved evidence sets, and use this score as a reward signal for RL fine-tuning. The result is a model that is incentivized to produce answers that are consistent with the evidence it was originally conditioned on. We demonstrate the approach on medical question answering and clinical decision support settings, including a cardiac disease dataset derived from MIMIC-IV-Ext, and report improvements in evidence alignment and faithfulness while preserving or improving task accuracy.

---

## 1. Introduction
Retrieval-augmented language models (RAG) improve factuality by grounding responses in external documents. However, RAG systems can still hallucinate or drift away from the retrieved evidence. This is especially risky in clinical settings, where a model’s response must align with trusted guidelines and patient context.

**EvidenceRL** addresses this gap by explicitly **rewarding evidence consistency**. We propose a reinforcement learning pipeline that compares evidence retrieved for the **input** (query + patient context) with evidence retrieved for the **output** (model response). The reward penalizes mismatches and encourages answers that are traceable to the original evidence.

Contributions:
1. A novel **evidence consistency reward** that compares input-time and output-time retrieved evidence.
2. An RL training pipeline that is compatible with standard RAG architectures and LLM generators.
3. A clinical evaluation setup that includes a prompting baseline, a RAG baseline, and evidence-consistency RL, with LLM-as-a-judge scoring for diagnoses and procedures.

---

## 2. Problem Setup
Let a user query (or patient case) be $x$, and a knowledge base be $K$. A standard RAG system retrieves top-$k$ documents:
$$E_\text{in} = \text{Retrieve}(x, K, k).$$
An LLM then generates an answer $y$ using $E_\text{in}$. After generation, we run a **second retrieval** over $K$ using the generated answer:
$$E_\text{out} = \text{Retrieve}(y, K, k).$$

We define an evidence consistency score $r_\text{evid}(E_\text{in}, E_\text{out}) \in [-1, 1]$ and use it as a reward for RL. Optionally, the reward is gated by correctness:
$$r = \mathbb{1}\{y \text{ is correct}\} \cdot r_\text{evid}.$$

---

## 3. Method: Evidence Consistency Reward
We compute evidence consistency using similarity between the input and output evidence sets. In the simplest case, we compare document embeddings with cosine similarity and aggregate across sets. A positive score indicates high overlap/consistency, while a negative score indicates divergence.

**Reward definition (illustrative):**
1. Retrieve evidence sets $E_\text{in}$ and $E_\text{out}$.
2. Compute a similarity $s(E_\text{in}, E_\text{out})$.
3. Map to a signed reward in $[-1, 1]$.

This reward encourages the model to produce answers that are aligned with the evidence it was originally conditioned on, and penalizes answers that “switch” evidence post-hoc.

---

## 4. RL Training Pipeline
We adopt an RL fine-tuning pipeline (e.g., PPO or policy gradient) over the generator:
1. **Retrieve** input evidence and generate answer.
2. **Retrieve** output evidence based on the generated answer.
3. **Compute reward** using evidence consistency (optionally gated by correctness).
4. **Update** the policy to maximize expected reward.

This pipeline is model-agnostic and works with any LLM that can be integrated into a RAG workflow.

---

## 5. Clinical Application: Cardiac Diagnosis and Procedures
We evaluate EvidenceRL on a MIMIC-IV-Ext cardiac dataset. Each patient case provides:
- **Context**: chief complaint, HPI, labs, ECG, imaging, etc.
- **Ground truth**: ICD diagnosis codes and procedure codes.

We define two tasks:
1. **Top-5 diagnoses prediction**
2. **Top-5 procedures prediction**

Baselines:
- **Prompting** (no external knowledge)
- **RAG** (retrieval from medical guideline documents)

EvidenceRL improves alignment between retrieved evidence and generated outputs, and can be combined with an LLM-as-a-judge to evaluate semantic correctness for precision/recall@k.

---

## 6. Experimental Design (Draft)
**Datasets**
- MIMIC-IV-Ext cardiac disease cohort
- Clinical guideline PDFs for RAG knowledge base

**Metrics**
- Evidence alignment score (input vs output evidence similarity)
- Precision/Recall@k for diagnoses and procedures (LLM-as-a-judge)
- Faithfulness assessment (e.g., evidence citation correctness)

**Ablations**
- Vary retrieval model strength
- Vary chunk sizes and overlap
- Reward gating on correctness vs no gating

---

## 7. Limitations and Risks
- Evidence similarity does not guarantee clinical correctness.
- LLM-as-a-judge introduces its own uncertainty and potential bias.
- Retrieval quality depends on chunking and embedding choice.
- Human evaluation is still required for high-stakes deployment.

---

## 8. Ethics and Responsible Use
EvidenceRL is intended to improve grounding and transparency in medical AI. It does **not** replace clinical judgment. All outputs should be validated by clinicians and used only in research or controlled decision-support contexts.

---

## 9. Conclusion
EvidenceRL provides a practical mechanism to enforce evidence consistency in RAG systems via reinforcement learning. The method is simple, model-agnostic, and well-suited for medical applications where grounding is critical. Future work will explore stronger evidence alignment objectives, improved retrieval strategies, and extensive human evaluation.

---

## Appendix A: Algorithm (Pseudo-code)
```
Input: query x, knowledge base K, retriever R, generator G
1. E_in = R(x, K, k)
2. y = G(x, E_in)
3. E_out = R(y, K, k)
4. r = evidence_consistency(E_in, E_out)
5. Update G to maximize r (RL step)
```
