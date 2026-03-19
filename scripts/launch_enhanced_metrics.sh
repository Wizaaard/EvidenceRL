#!/usr/bin/env bash
# ============================================================================
# EvidenceRL ENHANCED Metrics Launcher (with Judge Selection)
# ============================================================================
# Runs the enhanced metrics pipeline with:
# - All publication metrics (F1@k, Macro-F1, Micro-F1, retrieval, CIs)
# - Configurable judge model (try different judges!)
# - vLLM acceleration (14-24x faster)
#
# Usage:
#   ./launch_enhanced_metrics.sh gemma-3-27b                      # Default judge (medgemma)
#   ./launch_enhanced_metrics.sh gemma-3-27b --judge llama-3.1-8b # Custom judge
#   ./launch_enhanced_metrics.sh gemma-3-27b --judge-comparison    # Try multiple judges
#   ./launch_enhanced_metrics.sh --help
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Paths - ENHANCED METRICS directories (using absolute paths like baseline)
GENERATION_OUTPUT_DIR="${PROJECT_ROOT}/../generation_output"
METRICS_OUTPUT_DIR="${PROJECT_ROOT}/.."
# Configure these for your environment
MODEL_DIR="${MODEL_BASE_DIR:-//storage/ice-shared/bmed-sp-wang/Models}"
LOG_DIR="${PROJECT_ROOT}/logs_evidencerl"
CODE_DIR="${PROJECT_ROOT}"

# Default configuration
DEFAULT_JUDGE="medgemma-27b-it"
DEFAULT_TENSOR_PARALLEL=2
DEFAULT_GPU_MEM=0.90

# Recommended open-source judges (ordered by quality)
declare -A RECOMMENDED_JUDGES=(
    ["medgemma-27b-it"]="Medical specialist, best for diagnosis (default)"
    ["llama-3.1-8b"]="Fast, good general judge"
    ["llama-3.3-70b"]="Very accurate, slower"
    ["gemma-3-27b-it"]="Good balance of speed/quality"
    ["gpt-oss-120b"]="Large model, high quality"
    ["med42-70b"]="Medical specialist, very thorough"
)

# Map judge names to actual model directory names
declare -A JUDGE_MODEL_PATHS=(
    ["medgemma-27b-it"]="medgemma-27b-it"
    ["llama-3.1-8b"]="Llama-3.1-8B-Instruct"
    ["llama-3.3-70b"]="Llama-3.3-70B-Instruct"
    ["gemma-3-27b-it"]="gemma-3-27b-it"
    ["gpt-oss-120b"]="gpt-oss-120b"
    ["med42-70b"]="Llama3-Med42-70B"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    cat << EOF
${GREEN}EvidenceRL ENHANCED Metrics Launcher${NC}
${CYAN}Includes F1@k, Macro-F1, Retrieval metrics, Statistical CIs${NC}

Usage:
  $0 <generation_model> [options]

Arguments:
  generation_model    Model whose generation to evaluate (e.g., gemma-3-27b)

Options:
  --judge <model>              Judge model to use (default: ${DEFAULT_JUDGE})
  --nli-model <model>          NLI model for grounding (default: pritamdeka/PubMedBERT-MNLI-MedNLI)
  --judge-comparison           Run comparison with multiple judges
  --generation-file <path>     Explicit path to generation JSON
  --output-dir <path>          Output directory
  --tensor-parallel <n>        GPUs for judge (default: ${DEFAULT_TENSOR_PARALLEL})
  --gpu-mem <f>                GPU memory fraction (default: ${DEFAULT_GPU_MEM})
  --list-judges                List recommended judges
  --help                       Show this help

Recommended Judges:
EOF
    for judge in "${!RECOMMENDED_JUDGES[@]}"; do
        echo "  ${CYAN}${judge}${NC}: ${RECOMMENDED_JUDGES[$judge]}"
    done

    cat << EOF

Examples:
  # Use default judge (medgemma-27b-it)
  $0 gemma-3-27b

  # Try different judge
  $0 gemma-3-27b --judge llama-3.1-8b

  # Compare multiple judges
  $0 gemma-3-27b --judge-comparison

  # Explicit generation file path
  $0 --generation-file /path/to/gemma-3-27b_1000-v1.0.json --judge llama-3.3-70b

New Metrics Included:
  • F1@k (harmonic mean of P@k and R@k)
  • Macro-F1 and Micro-F1 (per-diagnosis classification)
  • Retrieval Success@k and MRR
  • 95% Confidence Intervals on ALL metrics
  • Per-diagnosis performance breakdown
EOF
}

list_judges() {
    echo -e "${GREEN}Recommended Open-Source LLM Judges${NC}"
    echo ""
    echo -e "${CYAN}Judge Selection Guide:${NC}"
    echo ""

    cat << EOF
${YELLOW}For Medical Diagnosis (Recommended):${NC}
  1. medgemma-27b-it      ⭐⭐⭐ Best for medical (default)
  2. med42-70b            ⭐⭐⭐ Most thorough, slower
  3. medgemma-4b-it       ⭐⭐  Fast, good quality

${YELLOW}For General/Fast Evaluation:${NC}
  4. llama-3.1-8b         ⭐⭐⭐ Fast, reliable
  5. gemma-3-27b-it       ⭐⭐⭐ Good balance
  6. gemma-3-12b-it       ⭐⭐  Faster, decent

${YELLOW}For Maximum Quality (Slow):${NC}
  7. llama-3.3-70b        ⭐⭐⭐ Very accurate
  8. gpt-oss-120b         ⭐⭐⭐ Highest quality
  9. med42-70b            ⭐⭐⭐ Medical specialist

${CYAN}Performance Comparison:${NC}
  Model              Speed    Quality   Medical   GPU Needed
  ──────────────────────────────────────────────────────────
  medgemma-4b        +++      ++        +++       2x H200
  llama-3.1-8b       +++      +++       +         2x H200
  gemma-3-27b        ++       +++       +         2x H200
  medgemma-27b       ++       +++       +++       2x H200 (default)
  llama-3.3-70b      +        +++       ++        4x H200
  med42-70b          +        +++       +++       4x H200
  gpt-oss-120b       +        +++       ++        4x H200

${CYAN}Recommendation:${NC}
  - For papers: Use ${GREEN}medgemma-27b-it${NC} (best medical + good speed)
  - To validate judge: Run ${GREEN}--judge-comparison${NC} to compare 3 judges
  - For speed: Use ${GREEN}llama-3.1-8b${NC}
  - For quality: Use ${GREEN}llama-3.3-70b${NC} or ${GREEN}med42-70b${NC}
EOF
}

find_generation_file() {
    local model_id="$1"
    local version="${2:-1.0}"

    # Common patterns (using wildcards for patient count flexibility)
    local patterns=(
        "${GENERATION_OUTPUT_DIR}/${model_id}_v${version}/${model_id}_*-v${version}.json"
        "${GENERATION_OUTPUT_DIR}/${model_id}/${model_id}_*-v${version}.json"
    )

    for pattern in "${patterns[@]}"; do
        if ls ${pattern} 1> /dev/null 2>&1; then
            echo $(ls ${pattern} | head -1)
            return 0
        fi
    done

    return 1
}

run_enhanced_metrics() {
    local generation_file="$1"
    local judge_model="$2"
    local output_dir="$3"
    local tensor_parallel="$4"
    local gpu_mem="$5"
    local nli_model="${6:-pritamdeka/PubMedBERT-MNLI-MedNLI}"
    local no_sentence_level="${7:-false}"

    echo -e "${CYAN}Submitting Enhanced Metrics Job...${NC}"
    echo "  Generation: ${generation_file}"
    echo "  Judge: ${judge_model}"
    echo "  NLI Model: ${nli_model}"
    echo "  Sentence-level NLI: $([ "${no_sentence_level}" = "true" ] && echo "disabled" || echo "enabled")"
    echo "  Output: ${output_dir}"
    echo "  GPUs: ${tensor_parallel}x H200"
    echo ""

    # Create output directory
    mkdir -p "${output_dir}"
    mkdir -p "${LOG_DIR}"

    # Determine judge model path (map judge name to actual model directory)
    local model_dir_name="${JUDGE_MODEL_PATHS[$judge_model]:-$judge_model}"
    local judge_path="${MODEL_DIR}/${model_dir_name}"

    # Output file
    local output_json="${output_dir}/enhanced_metrics.json"

    # Submit sbatch job
    local job_id
    job_id=$(sbatch --parsable \
        --gres=gpu:h200:${tensor_parallel} \
        --export=ALL,GENERATION_FILE="${generation_file}",JUDGE_MODEL="${judge_path}",NLI_MODEL="${nli_model}",OUTPUT_JSON="${output_json}",TENSOR_PARALLEL_SIZE="${tensor_parallel}",GPU_MEMORY_UTILIZATION="${gpu_mem}",CONDA_ENV="${CONDA_ENV:-ragcon}",NO_SENTENCE_LEVEL="${no_sentence_level}" \
        "${SCRIPT_DIR}/metrics_enhanced.sbatch")

    local submit_code=$?

    if [[ $submit_code -eq 0 ]]; then
        echo -e "${GREEN}✅ Job submitted: ${job_id}${NC}"
        echo ""
        echo "Monitor: tail -f ${LOG_DIR}/enhanced_metrics_${job_id}.out"
        echo "Results: ${output_json}"
        echo ""
        return 0
    else
        echo -e "${RED}❌ Job submission failed!${NC}"
        return $submit_code
    fi
}

run_judge_comparison() {
    local generation_file="$1"
    local base_output_dir="$2"

    # Judges to compare (fast, medium, slow)
    local judges_to_compare=(
        "llama-3.1-8b:Llama-3.1-8B-Instruct"
        "medgemma-27b:medgemma-27b-it"
        "gemma-3-27b:gemma-3-27b-it"
    )

    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Judge Comparison Mode${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Comparing ${#judges_to_compare[@]} judges:"
    for judge_spec in "${judges_to_compare[@]}"; do
        local judge_name="${judge_spec%%:*}"
        echo "  - ${judge_name}"
    done
    echo ""

    # Run each judge
    local results_files=()
    for judge_spec in "${judges_to_compare[@]}"; do
        local judge_name="${judge_spec%%:*}"
        local judge_model="${judge_spec##*:}"

        echo -e "${YELLOW}Running with judge: ${judge_name}${NC}"

        local judge_output_dir="${base_output_dir}/judge_${judge_name}"

        # Determine tensor parallel size (70B models need 4 GPUs)
        local tensor_parallel=2
        if [[ "${judge_model}" == *"70"* ]] || [[ "${judge_model}" == *"120"* ]]; then
            tensor_parallel=4
        fi

        if run_enhanced_metrics \
            "${generation_file}" \
            "${judge_model}" \
            "${judge_output_dir}" \
            "${tensor_parallel}" \
            "${DEFAULT_GPU_MEM}"; then

            results_files+=("${judge_name}:${judge_output_dir}/enhanced_metrics.json")
        fi

        echo ""
    done

    # Compare results
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Judge Comparison Results${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo ""

    if command -v jq &> /dev/null; then
        echo "Metric Comparison:"
        echo ""
        printf "%-20s %-10s %-10s %-10s %-10s\n" "Judge" "P@3" "F1@3" "Macro-F1" "Reward@3"
        echo "────────────────────────────────────────────────────────────────"

        for result_spec in "${results_files[@]}"; do
            local judge_name="${result_spec%%:*}"
            local metrics_file="${result_spec##*:}"

            if [[ -f "${metrics_file}" ]]; then
                local p3=$(jq -r '.aggregate_metrics.avg_precision_at_k."3"' "${metrics_file}")
                local f13=$(jq -r '.aggregate_metrics.avg_f1_at_k."3"' "${metrics_file}")
                local macro=$(jq -r '.aggregate_metrics.macro_f1' "${metrics_file}")
                local reward=$(jq -r '.aggregate_metrics.avg_reward_combined_at_k."3"' "${metrics_file}")

                printf "%-20s %-10.3f %-10.3f %-10.3f %-10.3f\n" \
                    "${judge_name}" "${p3}" "${f13}" "${macro}" "${reward}"
            fi
        done

        echo ""
        echo -e "${YELLOW}Note:${NC} Different judges may have different strictness levels."
        echo "      Choose the judge that best aligns with human evaluation."
    fi

    # Save comparison report
    local comparison_file="${base_output_dir}/judge_comparison.txt"
    echo "Judge Comparison Report" > "${comparison_file}"
    echo "Generation: ${generation_file}" >> "${comparison_file}"
    echo "" >> "${comparison_file}"

    for result_spec in "${results_files[@]}"; do
        local judge_name="${result_spec%%:*}"
        local metrics_file="${result_spec##*:}"
        echo "${judge_name}: ${metrics_file}" >> "${comparison_file}"
    done

    echo -e "${GREEN}Comparison saved to: ${comparison_file}${NC}"
}

# Main
main() {
    # Parse arguments
    local generation_model=""
    local judge_model="${DEFAULT_JUDGE}"
    local nli_model="pritamdeka/PubMedBERT-MNLI-MedNLI"
    local generation_file=""
    local output_dir=""
    local tensor_parallel="${DEFAULT_TENSOR_PARALLEL}"
    local gpu_mem="${DEFAULT_GPU_MEM}"
    local judge_comparison=false
    local no_sentence_level=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                show_help
                exit 0
                ;;
            --list-judges)
                list_judges
                exit 0
                ;;
            --judge)
                judge_model="$2"
                shift 2
                ;;
            --judge-comparison)
                judge_comparison=true
                shift
                ;;
            --generation-file)
                generation_file="$2"
                shift 2
                ;;
            --output-dir)
                output_dir="$2"
                shift 2
                ;;
            --tensor-parallel)
                tensor_parallel="$2"
                shift 2
                ;;
            --gpu-mem)
                gpu_mem="$2"
                shift 2
                ;;
            --nli-model)
                nli_model="$2"
                shift 2
                ;;
            --no-sentence-level)
                no_sentence_level=true
                shift
                ;;
            -*)
                echo -e "${RED}Unknown option: $1${NC}"
                show_help
                exit 1
                ;;
            *)
                generation_model="$1"
                shift
                ;;
        esac
    done

    # Find generation file if not specified
    if [[ -z "${generation_file}" ]]; then
        if [[ -z "${generation_model}" ]]; then
            echo -e "${RED}Error: Must specify generation_model or --generation-file${NC}"
            show_help
            exit 1
        fi

        if ! generation_file=$(find_generation_file "${generation_model}"); then
            echo -e "${RED}Error: Could not find generation file for ${generation_model}${NC}"
            echo "Looked for patterns:"
            echo "  generation_output/${generation_model}_v1.0/${generation_model}_1000-v1.0.json"
            echo "  generation_output/${generation_model}/${generation_model}_1000-v1.0.json"
            exit 1
        fi
    fi

    # Verify generation file exists and convert to absolute path
    if [[ ! -f "${generation_file}" ]]; then
        echo -e "${RED}Error: Generation file not found: ${generation_file}${NC}"
        exit 1
    fi

    # Convert to absolute path for sbatch
    generation_file="$(cd "$(dirname "${generation_file}")" && pwd)/$(basename "${generation_file}")"

    # Set default output directory
    if [[ -z "${output_dir}" ]]; then
        local model_name=$(basename $(dirname "${generation_file}"))
        output_dir="${METRICS_OUTPUT_DIR}/${model_name}_enhanced"
    else
        # Convert relative output_dir to absolute path
        if [[ "${output_dir}" != /* ]]; then
            output_dir="${METRICS_OUTPUT_DIR}/${output_dir}"
        fi
    fi

    # Run metrics
    if [[ "${judge_comparison}" == true ]]; then
        run_judge_comparison "${generation_file}" "${output_dir}"
    else
        run_enhanced_metrics \
            "${generation_file}" \
            "${judge_model}" \
            "${output_dir}" \
            "${tensor_parallel}" \
            "${gpu_mem}" \
            "${nli_model}" \
            "${no_sentence_level}"
    fi
}

main "$@"
