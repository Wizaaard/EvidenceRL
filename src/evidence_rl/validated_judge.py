"""
Validated Judge with Confidence Scoring and Calibration Metrics

Extends the LLM judge to provide confidence scores and validation capabilities.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass, field
import re
import json


@dataclass
class JudgeVerdict:
    """Result of judge evaluation with confidence."""
    is_correct: bool
    confidence: float  # 0.0 to 1.0
    raw_output: str
    requires_review: bool = False


@dataclass
class CalibrationMetrics:
    """Expected Calibration Error and related metrics."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    accuracy: float  # Overall accuracy
    avg_confidence: float  # Average confidence
    calibration_curve: Dict[str, List[float]] = field(default_factory=dict)


class ValidatedJudge:
    """
    Enhanced LLM Judge with confidence scoring and validation.

    This wrapper adds:
    - Confidence extraction from judge outputs
    - Calibration metrics computation
    - Low-confidence case flagging for manual review
    - Inter-rater agreement analysis
    """

    def __init__(self, base_judge, confidence_threshold: float = 0.6):
        """
        Initialize validated judge.

        Args:
            base_judge: The underlying LLMAnswerJudge or VLLMAnswerJudge
            confidence_threshold: Threshold below which cases need review
        """
        self.base_judge = base_judge
        self.confidence_threshold = confidence_threshold
        self.low_confidence_cases = []
        self.evaluation_history = []

    def evaluate_with_confidence(
        self,
        diagnosis_name: str,
        ground_truths: List[str]
    ) -> JudgeVerdict:
        """
        Evaluate diagnosis with confidence scoring.

        Args:
            diagnosis_name: The predicted diagnosis
            ground_truths: List of accepted ground truth diagnoses

        Returns:
            JudgeVerdict with correctness, confidence, and review flag
        """
        # Build enhanced prompt that asks for confidence
        enhanced_prompt = self._build_confidence_prompt(diagnosis_name, ground_truths)

        # Get verdict from base judge
        raw_output = self._get_raw_output(enhanced_prompt)

        # Parse verdict and confidence
        is_correct, confidence = self._parse_verdict_with_confidence(raw_output)

        # Create verdict object
        verdict = JudgeVerdict(
            is_correct=is_correct,
            confidence=confidence,
            raw_output=raw_output,
            requires_review=(confidence < self.confidence_threshold)
        )

        # Log for calibration analysis
        self.evaluation_history.append({
            "diagnosis": diagnosis_name,
            "ground_truths": ground_truths,
            "is_correct": is_correct,
            "confidence": confidence
        })

        # Flag low-confidence cases
        if verdict.requires_review:
            self.low_confidence_cases.append({
                "diagnosis": diagnosis_name,
                "ground_truths": ground_truths,
                "confidence": confidence,
                "verdict": is_correct
            })

        return verdict

    def _build_confidence_prompt(self, diagnosis: str, ground_truths: List[str]) -> str:
        """Build prompt that asks for confidence scoring."""
        gt_str = "\n".join([f"- {gt}" for gt in ground_truths])

        return f"""You are evaluating the diagnosis prediction of a clinical model.

CANDIDATE ANSWER: "{diagnosis}"

ACCEPTED GROUND TRUTHS:
{gt_str}

TASK: Does the CANDIDATE ANSWER semantically match ANY of the ACCEPTED GROUND TRUTHS?

Consider:
- Synonyms (e.g., "CHF" = "Congestive Heart Failure")
- Abbreviations (e.g., "MI" = "Myocardial Infarction")
- Minor wording differences (e.g., "Acute heart failure" ≈ "Acute cardiac failure")

RESPOND IN THIS FORMAT:
Verdict: TRUE or FALSE
Confidence: [LOW/MEDIUM/HIGH] or [0.0-1.0]
Reasoning: [Brief explanation]

Examples:
- Candidate="Heart failure", Truth="Congestive heart failure" → Verdict: TRUE, Confidence: HIGH
- Candidate="Myocardial infarction", Truth="Heart failure" → Verdict: FALSE, Confidence: HIGH
- Candidate="Cardiac arrest", Truth="Heart attack" → Verdict: FALSE, Confidence: MEDIUM (similar but distinct)

Your evaluation:
"""

    def _get_raw_output(self, prompt: str) -> str:
        """Get raw output from base judge."""
        try:
            # Check if base judge has a method to return raw output
            if hasattr(self.base_judge, '_pipeline'):
                # HuggingFace generator
                result = self.base_judge._pipeline(
                    prompt,
                    max_new_tokens=150,
                    do_sample=False,
                    temperature=0.0
                )
                return result[0]['generated_text'] if isinstance(result, list) else result
            elif hasattr(self.base_judge, '_llm'):
                # vLLM generator
                from vllm import SamplingParams
                params = SamplingParams(
                    max_tokens=150,
                    temperature=0.0,
                    top_p=1.0
                )
                result = self.base_judge._llm.generate([prompt], params)
                return result[0].outputs[0].text
            else:
                # Fallback: use base judge's is_correct method
                # (won't have confidence, but will work)
                is_correct = self.base_judge.is_correct(prompt)
                return f"Verdict: {'TRUE' if is_correct else 'FALSE'}\nConfidence: MEDIUM"
        except Exception as e:
            print(f"Warning: Judge evaluation failed: {e}")
            return "Verdict: FALSE\nConfidence: LOW\nReasoning: Evaluation error"

    def _parse_verdict_with_confidence(self, raw_output: str) -> Tuple[bool, float]:
        """
        Parse verdict and confidence from judge output.

        Returns:
            (is_correct, confidence_score)
        """
        text = raw_output.strip().lower()

        # Extract verdict
        is_correct = False
        if "verdict:" in text:
            verdict_match = re.search(r"verdict:\s*(true|false)", text)
            if verdict_match:
                is_correct = verdict_match.group(1) == "true"
        elif text.startswith("true"):
            is_correct = True
        elif text.startswith("false"):
            is_correct = False

        # Extract confidence
        confidence = 0.5  # Default medium confidence

        # Look for explicit confidence score
        conf_match = re.search(r"confidence:\s*([0-9.]+)", text)
        if conf_match:
            confidence = float(conf_match.group(1))
        else:
            # Look for textual confidence
            if "high" in text:
                confidence = 0.9
            elif "medium" in text:
                confidence = 0.6
            elif "low" in text:
                confidence = 0.3

        return is_correct, confidence

    def compute_calibration_metrics(
        self,
        gold_standard: Optional[List[bool]] = None
    ) -> CalibrationMetrics:
        """
        Compute Expected Calibration Error and related metrics.

        Args:
            gold_standard: Optional list of ground truth verdicts.
                          If None, uses judge's own verdicts (self-calibration)

        Returns:
            CalibrationMetrics with ECE, MCE, and calibration curve
        """
        if not self.evaluation_history:
            raise ValueError("No evaluation history available for calibration")

        verdicts = np.array([h["is_correct"] for h in self.evaluation_history])
        confidences = np.array([h["confidence"] for h in self.evaluation_history])

        # Use gold standard if provided, else use verdicts as ground truth
        ground_truth = np.array(gold_standard) if gold_standard else verdicts

        # Compute calibration in bins
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        ece = 0.0  # Expected Calibration Error
        mce = 0.0  # Maximum Calibration Error

        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]

            # Find samples in this bin
            in_bin = (confidences >= lower) & (confidences < upper)
            if i == n_bins - 1:  # Include upper boundary in last bin
                in_bin |= (confidences == 1.0)

            n_in_bin = in_bin.sum()

            if n_in_bin > 0:
                bin_accuracy = ground_truth[in_bin].mean()
                bin_confidence = confidences[in_bin].mean()

                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
                bin_counts.append(n_in_bin)

                # Update ECE (weighted by bin size)
                ece += (n_in_bin / len(confidences)) * abs(bin_accuracy - bin_confidence)

                # Update MCE
                mce = max(mce, abs(bin_accuracy - bin_confidence))
            else:
                bin_accuracies.append(np.nan)
                bin_confidences.append(np.nan)
                bin_counts.append(0)

        return CalibrationMetrics(
            ece=float(ece),
            mce=float(mce),
            accuracy=float(ground_truth.mean()),
            avg_confidence=float(confidences.mean()),
            calibration_curve={
                "bin_confidences": bin_confidences,
                "bin_accuracies": bin_accuracies,
                "bin_counts": bin_counts
            }
        )

    def compute_inter_rater_agreement(
        self,
        expert_verdicts: List[bool]
    ) -> Dict[str, float]:
        """
        Compute agreement between judge and expert raters.

        Args:
            expert_verdicts: List of verdicts from expert human raters

        Returns:
            Dictionary with agreement metrics (accuracy, kappa, F1)
        """
        if len(expert_verdicts) != len(self.evaluation_history):
            raise ValueError("Expert verdicts must match evaluation history length")

        judge_verdicts = np.array([h["is_correct"] for h in self.evaluation_history])
        expert_verdicts = np.array(expert_verdicts)

        # Simple accuracy
        accuracy = (judge_verdicts == expert_verdicts).mean()

        # Cohen's kappa (chance-adjusted agreement)
        n = len(judge_verdicts)
        p_o = accuracy  # Observed agreement

        # Expected agreement by chance
        p_judge_true = judge_verdicts.mean()
        p_expert_true = expert_verdicts.mean()
        p_e = (p_judge_true * p_expert_true +
               (1 - p_judge_true) * (1 - p_expert_true))

        kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0

        # F1 score
        tp = ((judge_verdicts == 1) & (expert_verdicts == 1)).sum()
        fp = ((judge_verdicts == 1) & (expert_verdicts == 0)).sum()
        fn = ((judge_verdicts == 0) & (expert_verdicts == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "agreement": float(accuracy),
            "cohens_kappa": float(kappa),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

    def get_low_confidence_cases(self, limit: int = 10) -> List[Dict]:
        """
        Get cases that need manual review.

        Args:
            limit: Maximum number of cases to return

        Returns:
            List of low-confidence cases for review
        """
        # Sort by confidence (lowest first)
        sorted_cases = sorted(
            self.low_confidence_cases,
            key=lambda x: x["confidence"]
        )
        return sorted_cases[:limit]

    def export_for_annotation(self, output_path: str):
        """
        Export low-confidence cases for human annotation.

        Args:
            output_path: Path to save JSON file for annotation
        """
        annotation_data = {
            "instructions": (
                "For each case, review the diagnosis and ground truths, "
                "then mark is_correct as true or false."
            ),
            "cases": [
                {
                    "id": i,
                    "diagnosis": case["diagnosis"],
                    "ground_truths": case["ground_truths"],
                    "judge_verdict": case["verdict"],
                    "judge_confidence": case["confidence"],
                    "is_correct": None,  # To be filled by human
                    "notes": ""  # Optional notes
                }
                for i, case in enumerate(self.low_confidence_cases)
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(annotation_data, f, indent=2)

        print(f"Exported {len(self.low_confidence_cases)} cases to {output_path}")


if __name__ == "__main__":
    # Example usage (simulation)
    print("=== Validated Judge Demo ===\n")

    # Simulate a base judge (mock)
    class MockJudge:
        def _pipeline(self, prompt, **kwargs):
            # Simulate some verdicts
            if "heart failure" in prompt.lower() and "chf" in prompt.lower():
                return "Verdict: TRUE\nConfidence: HIGH\nReasoning: CHF is congestive heart failure"
            else:
                return "Verdict: FALSE\nConfidence: LOW\nReasoning: Different conditions"

    base_judge = MockJudge()
    validated_judge = ValidatedJudge(base_judge, confidence_threshold=0.6)

    # Test evaluation
    verdict1 = validated_judge.evaluate_with_confidence(
        "Heart failure",
        ["Congestive heart failure", "CHF"]
    )
    print(f"Verdict 1: {verdict1.is_correct}, Confidence: {verdict1.confidence:.2f}")

    verdict2 = validated_judge.evaluate_with_confidence(
        "Myocardial infarction",
        ["Heart failure"]
    )
    print(f"Verdict 2: {verdict2.is_correct}, Confidence: {verdict2.confidence:.2f}")

    print(f"\nLow confidence cases: {len(validated_judge.low_confidence_cases)}")
