"""
Parse Error Classification and Analysis

Provides detailed error tracking for JSON parsing failures to identify
systematic issues with model outputs.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import json
import re


class ParsingErrorType(Enum):
    """Classification of parsing error types."""
    JSON_SYNTAX = "json_syntax"  # Invalid JSON syntax
    MISSING_FIELDS = "missing_fields"  # Required fields missing
    TYPE_MISMATCH = "type_mismatch"  # Wrong data types
    TRUNCATION = "truncation"  # Output appears truncated
    ENCODING = "encoding"  # Unicode/encoding issues
    INCOMPLETE_JSON = "incomplete_json"  # Unclosed brackets/quotes
    EXTRA_TEXT = "extra_text"  # Text before/after JSON
    EMPTY_OUTPUT = "empty_output"  # No output generated
    UNKNOWN = "unknown"  # Other errors


@dataclass
class ParsingError:
    """Detailed information about a parsing failure."""
    error_type: ParsingErrorType
    error_message: str
    raw_output: str
    raw_output_length: int
    hadm_id: Optional[str] = None
    detected_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "error_type": self.error_type.value,
            "error_message": self.error_message,
            "raw_output_preview": self.raw_output[:500] + "..." if len(self.raw_output) > 500 else self.raw_output,
            "raw_output_length": self.raw_output_length,
            "hadm_id": self.hadm_id,
            "detected_patterns": self.detected_patterns
        }


@dataclass
class ParsingErrorSummary:
    """Summary statistics of parsing errors."""
    total_attempts: int
    total_failures: int
    failure_rate: float
    error_type_distribution: Dict[str, int]
    error_type_percentages: Dict[str, float]
    example_errors: Dict[str, List[Dict]]  # Type -> list of examples

    def __str__(self):
        lines = [
            f"=== Parsing Error Summary ===",
            f"Total attempts: {self.total_attempts}",
            f"Total failures: {self.total_failures}",
            f"Failure rate: {self.failure_rate:.2%}",
            f"\nError Type Distribution:"
        ]

        for error_type, count in sorted(
            self.error_type_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            pct = self.error_type_percentages[error_type]
            lines.append(f"  {error_type}: {count} ({pct:.1%})")

        return "\n".join(lines)


class ParseErrorAnalyzer:
    """
    Analyzer for tracking and classifying parsing errors.

    Usage:
        analyzer = ParseErrorAnalyzer()

        # Track errors
        analyzer.track_attempt(raw_output, error, hadm_id)

        # Get summary
        summary = analyzer.get_summary()
        print(summary)

        # Export for debugging
        analyzer.export_error_examples("parse_errors.json")
    """

    def __init__(self, max_examples_per_type: int = 5):
        """
        Initialize error analyzer.

        Args:
            max_examples_per_type: Maximum examples to store per error type
        """
        self.max_examples = max_examples_per_type
        self.errors: List[ParsingError] = []
        self.total_attempts = 0
        self.successful_parses = 0

    def track_attempt(
        self,
        raw_output: str,
        error: Optional[Exception] = None,
        hadm_id: Optional[str] = None,
        success: bool = False
    ):
        """
        Track a parsing attempt.

        Args:
            raw_output: The raw model output
            error: The exception raised (if any)
            hadm_id: Hospital admission ID for tracking
            success: Whether parsing succeeded
        """
        self.total_attempts += 1

        if success:
            self.successful_parses += 1
            return

        # Classify the error
        error_type, error_message, patterns = self._classify_error(raw_output, error)

        # Create error record
        parsing_error = ParsingError(
            error_type=error_type,
            error_message=error_message,
            raw_output=raw_output,
            raw_output_length=len(raw_output),
            hadm_id=hadm_id,
            detected_patterns=patterns
        )

        self.errors.append(parsing_error)

    def _classify_error(
        self,
        raw_output: str,
        error: Optional[Exception]
    ) -> Tuple[ParsingErrorType, str, List[str]]:
        """
        Classify the error type based on output and exception.

        Returns:
            (error_type, error_message, detected_patterns)
        """
        patterns = []

        # Check for empty output
        if not raw_output or len(raw_output.strip()) == 0:
            return ParsingErrorType.EMPTY_OUTPUT, "No output generated", patterns

        # Check for truncation indicators
        truncation_indicators = [
            raw_output.endswith("..."),
            raw_output.endswith(".."),
            not raw_output.rstrip().endswith("}"),
            not raw_output.rstrip().endswith("]")
        ]
        if any(truncation_indicators):
            patterns.append("appears_truncated")

        # Check for incomplete JSON
        open_braces = raw_output.count("{")
        close_braces = raw_output.count("}")
        open_brackets = raw_output.count("[")
        close_brackets = raw_output.count("]")

        if open_braces != close_braces:
            patterns.append(f"unmatched_braces({open_braces}≠{close_braces})")
            return (
                ParsingErrorType.INCOMPLETE_JSON,
                f"Unmatched braces: {open_braces} open, {close_braces} close",
                patterns
            )

        if open_brackets != close_brackets:
            patterns.append(f"unmatched_brackets({open_brackets}≠{close_brackets})")
            return (
                ParsingErrorType.INCOMPLETE_JSON,
                f"Unmatched brackets: {open_brackets} open, {close_brackets} close",
                patterns
            )

        # Check for text before/after JSON
        stripped = raw_output.strip()
        if not stripped.startswith("{") and not stripped.startswith("["):
            patterns.append("extra_text_before")
            # Try to find JSON start
            json_start = max(stripped.find("{"), stripped.find("["))
            if json_start > 0:
                return (
                    ParsingErrorType.EXTRA_TEXT,
                    f"Found {json_start} characters before JSON start",
                    patterns
                )

        # Check for encoding issues
        if "\\u" in raw_output or "\\x" in raw_output:
            patterns.append("escape_sequences")

        # Invalid control characters
        if any(ord(c) < 32 and c not in "\n\r\t" for c in raw_output):
            patterns.append("control_characters")
            return (
                ParsingErrorType.ENCODING,
                "Invalid control characters found",
                patterns
            )

        # Check error message
        if error:
            error_str = str(error).lower()

            if "json" in error_str and "decode" in error_str:
                # Try to extract specific JSON syntax errors
                if "expecting" in error_str:
                    patterns.append("json_syntax_expecting")
                    return (
                        ParsingErrorType.JSON_SYNTAX,
                        str(error),
                        patterns
                    )
                elif "extra data" in error_str:
                    patterns.append("json_extra_data")
                    return (
                        ParsingErrorType.JSON_SYNTAX,
                        "Extra data after JSON",
                        patterns
                    )

            if "key" in error_str or "field" in error_str:
                patterns.append("missing_field")
                return (
                    ParsingErrorType.MISSING_FIELDS,
                    str(error),
                    patterns
                )

            if "type" in error_str:
                patterns.append("type_error")
                return (
                    ParsingErrorType.TYPE_MISMATCH,
                    str(error),
                    patterns
                )

        # Default to JSON syntax error if we have an error
        if error:
            return (
                ParsingErrorType.JSON_SYNTAX,
                str(error),
                patterns
            )

        # Unknown if no error object
        return (
            ParsingErrorType.UNKNOWN,
            "Parsing failed without specific error",
            patterns
        )

    def get_summary(self) -> ParsingErrorSummary:
        """
        Get summary statistics of parsing errors.

        Returns:
            ParsingErrorSummary with distribution and examples
        """
        if self.total_attempts == 0:
            return ParsingErrorSummary(
                total_attempts=0,
                total_failures=0,
                failure_rate=0.0,
                error_type_distribution={},
                error_type_percentages={},
                example_errors={}
            )

        # Count error types
        error_type_counts = Counter(e.error_type for e in self.errors)
        error_type_distribution = {
            et.value: error_type_counts[et]
            for et in ParsingErrorType
            if error_type_counts[et] > 0
        }

        # Compute percentages
        total_errors = len(self.errors)
        error_type_percentages = {
            et: count / total_errors
            for et, count in error_type_distribution.items()
        }

        # Collect examples
        example_errors = {}
        for error_type in ParsingErrorType:
            examples = [
                e.to_dict()
                for e in self.errors
                if e.error_type == error_type
            ][:self.max_examples]

            if examples:
                example_errors[error_type.value] = examples

        return ParsingErrorSummary(
            total_attempts=self.total_attempts,
            total_failures=total_errors,
            failure_rate=total_errors / self.total_attempts,
            error_type_distribution=error_type_distribution,
            error_type_percentages=error_type_percentages,
            example_errors=example_errors
        )

    def export_error_examples(self, output_path: str):
        """
        Export error examples to JSON file for debugging.

        Args:
            output_path: Path to save JSON file
        """
        summary = self.get_summary()

        export_data = {
            "summary": {
                "total_attempts": summary.total_attempts,
                "total_failures": summary.total_failures,
                "failure_rate": summary.failure_rate,
                "error_type_distribution": summary.error_type_distribution,
                "error_type_percentages": summary.error_type_percentages
            },
            "examples_by_type": summary.example_errors
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Exported error analysis to {output_path}")

    def get_recommendations(self) -> List[str]:
        """
        Get actionable recommendations based on error patterns.

        Returns:
            List of recommendation strings
        """
        recommendations = []
        summary = self.get_summary()

        if summary.total_failures == 0:
            return ["No parsing failures detected - system is working well!"]

        dist = summary.error_type_distribution

        # High truncation rate
        if dist.get("truncation", 0) / summary.total_failures > 0.3:
            recommendations.append(
                "⚠️ HIGH TRUNCATION RATE (>30%): Increase max_tokens in generation config"
            )

        # JSON syntax errors
        if dist.get("json_syntax", 0) / summary.total_failures > 0.3:
            recommendations.append(
                "⚠️ HIGH JSON SYNTAX ERRORS: Consider using structured output mode "
                "(e.g., vLLM guided generation) or improve prompt with more examples"
            )

        # Incomplete JSON
        if dist.get("incomplete_json", 0) / summary.total_failures > 0.2:
            recommendations.append(
                "⚠️ INCOMPLETE JSON: Model may be hitting token limits or generating "
                "invalid structure. Add explicit JSON structure to system prompt."
            )

        # Extra text
        if dist.get("extra_text", 0) / summary.total_failures > 0.1:
            recommendations.append(
                "💡 EXTRA TEXT BEFORE JSON: Model is adding preamble. "
                "Update prompt to say 'Output ONLY the JSON, no other text.'"
            )

        # Missing fields
        if dist.get("missing_fields", 0) / summary.total_failures > 0.1:
            recommendations.append(
                "💡 MISSING FIELDS: Add field validation in prompt with explicit examples"
            )

        # Empty output
        if dist.get("empty_output", 0) / summary.total_failures > 0.05:
            recommendations.append(
                "⚠️ EMPTY OUTPUTS: Check inference pipeline - model may be failing silently"
            )

        # General high failure rate
        if summary.failure_rate > 0.5:
            recommendations.append(
                "🚨 FAILURE RATE >50%: Consider switching to a more capable model "
                "or using structured output constraints"
            )

        return recommendations if recommendations else [
            "✓ Error patterns are distributed - no single dominant failure mode"
        ]


if __name__ == "__main__":
    # Demo usage
    analyzer = ParseErrorAnalyzer()

    # Simulate some parsing attempts
    test_cases = [
        ('{"diagnoses": [{"name": "Test"}]}', None, True),  # Success
        ('{"diagnoses": [{"name": "Test"', ValueError("Expecting '}' at char 30"), False),  # Incomplete
        ('Here is the output: {"diagnoses": []}', ValueError("Extra data"), False),  # Extra text
        ('', None, False),  # Empty
        ('{"diagnoses": [{"name": "Test"}]}', None, True),  # Success
    ]

    for raw, error, success in test_cases:
        analyzer.track_attempt(raw, error, hadm_id="test", success=success)

    # Print summary
    summary = analyzer.get_summary()
    print(summary)
    print("\n" + "="*50)
    print("Recommendations:")
    for rec in analyzer.get_recommendations():
        print(f"  {rec}")
