#!/usr/bin/env bash
#SBATCH -J erl_sft_vllm_master
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH -o logs_evidencerl_sft/generation_sft_vllm_master_%j.out
#SBATCH -e logs_evidencerl_sft/generation_sft_vllm_master_%j.err

# ============================================================================
# EvidenceRL SFT Generation Master (vLLM Version)
# ============================================================================
# Orchestrates distributed SFT LoRA generation (no-RAG) using vLLM workers.
#
# Required environment variables:
#   MODEL_ID         - Short identifier for the SFT model
#   BASE_MODEL_PATH  - Full path to the base model
#   LORA_PATH        - Full path to the LoRA adapter directory
#   VERSION          - Version string for output naming
#
# Optional environment variables:
#   NUM_WORKERS      - Number of worker jobs (default: 5)
#   TOTAL_PATIENTS   - Total patients to process (default: 1000)
#   PATIENT_OFFSET   - Starting patient index (default: 0)
#   NUM_GPUS         - GPUs per worker (default: 2)
#   EXTRACTOR_MODEL  - Path to separate model for fallback extraction
#   MAX_TOKENS       - Max generation tokens (default: 2048)
# ============================================================================

set -euo pipefail

# Validate required environment variables
: "${MODEL_ID:?Need MODEL_ID}"
: "${BASE_MODEL_PATH:?Need BASE_MODEL_PATH}"
: "${LORA_PATH:?Need LORA_PATH}"
: "${VERSION:=1.0}"

# Configuration
NUM_WORKERS="${NUM_WORKERS:-5}"
TOTAL_PATIENTS="${TOTAL_PATIENTS:-1000}"
PATIENT_OFFSET="${PATIENT_OFFSET:-0}"
MAX_RETRIES="${MAX_RETRIES:-3}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"

# Paths
PROJECT_ROOT="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code"
OUTPUT_BASE_DIR="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/generation_sft_output/${MODEL_ID}_v${VERSION}"
WORKER_SCRIPT="${PROJECT_ROOT}/scripts/generation_sft_vllm/generation_sft_vllm_worker.sbatch"
MERGE_SCRIPT="${PROJECT_ROOT}/scripts/generation/merge_generation_results.py"

# Resource configuration
NUM_GPUS="${NUM_GPUS:-2}"
WORKER_CPUS="${WORKER_CPUS:-16}"
WORKER_MEM="${WORKER_MEM:-128G}"
WORKER_TIME="${WORKER_TIME:-02:00:00}"
MAX_TOKENS="${MAX_TOKENS:-2048}"

# Calculate patients per worker
PATIENTS_PER_WORKER=$(( (TOTAL_PATIENTS + NUM_WORKERS - 1) / NUM_WORKERS ))

# Create directories
SIGNAL_DIR="${OUTPUT_BASE_DIR}/signals"
WORKER_OUTPUT_DIR="${OUTPUT_BASE_DIR}/workers"
mkdir -p "${OUTPUT_BASE_DIR}" "${SIGNAL_DIR}" "${WORKER_OUTPUT_DIR}" "${PROJECT_ROOT}/logs_evidencerl_sft"

echo "============================================================"
echo "EvidenceRL SFT Generation Master (vLLM)"
echo "============================================================"
echo "MODE:            SFT LoRA (no-RAG)"
echo "MODEL_ID:        ${MODEL_ID}"
echo "BASE_MODEL:      ${BASE_MODEL_PATH}"
echo "LORA_PATH:       ${LORA_PATH}"
echo "VERSION:         ${VERSION}"
echo "OUTPUT_DIR:      ${OUTPUT_BASE_DIR}"
echo "PATIENT_RANGE:   ${PATIENT_OFFSET} to $((PATIENT_OFFSET + TOTAL_PATIENTS)) (${TOTAL_PATIENTS} patients)"
echo "NUM_WORKERS:     ${NUM_WORKERS}"
echo "PATIENTS/WORKER: ~${PATIENTS_PER_WORKER}"
echo "NUM_GPUS:        ${NUM_GPUS}"
echo "MAX_TOKENS:      ${MAX_TOKENS}"
echo "EXTRACTOR:       ${EXTRACTOR_MODEL:-same as generator}"
echo "Inference:       vLLM + LoRA (no-RAG)"
echo "============================================================"

# Initialize conda
CONDA_INIT_SCRIPT=""
for conda_path in \
    "$HOME/.conda/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh"; do
    if [ -f "$conda_path" ]; then
        CONDA_INIT_SCRIPT="$conda_path"
        break
    fi
done

if [ -n "$CONDA_INIT_SCRIPT" ]; then
    source "$CONDA_INIT_SCRIPT"
else
    CONDA_EXE_PATH="$(command -v conda 2>/dev/null || true)"
    if [ -n "$CONDA_EXE_PATH" ] && [ -x "$CONDA_EXE_PATH" ]; then
        CONDA_BASE="$("$CONDA_EXE_PATH" info --base)"
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    fi
fi

CONDA_ENV="${CONDA_ENV:-evidencerl}"
set +u
conda activate "${CONDA_ENV}" 2>/dev/null || true
set -u

# Explicitly set CONDA_PREFIX if conda activate didn't set it
if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "${CONDA_PREFIX}" ]; then
    for conda_base in \
        "$HOME/scratch/conda_envs" \
        "$HOME/.conda/envs" \
        "$HOME/miniconda3/envs" \
        "$HOME/anaconda3/envs" \
        "/storage/ice1/8/6/$USER/conda_envs"; do
        if [ -d "${conda_base}/${CONDA_ENV}" ]; then
            export CONDA_PREFIX="${conda_base}/${CONDA_ENV}"
            echo "[sft master] Set CONDA_PREFIX to: ${CONDA_PREFIX}"
            break
        fi
    done
fi

# Verify CONDA_PREFIX is set and valid
if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "${CONDA_PREFIX}" ]; then
    echo "[ERROR] Could not locate conda environment: ${CONDA_ENV}"
    echo "[ERROR] CONDA_PREFIX=${CONDA_PREFIX:-not set}"
    exit 2
fi

echo "[sft master] Using conda env: ${CONDA_ENV} at ${CONDA_PREFIX}"

# Verify python executable exists
if [ ! -x "${CONDA_PREFIX}/bin/python" ]; then
    echo "[ERROR] Python executable not found at: ${CONDA_PREFIX}/bin/python"
    exit 2
fi

# Track worker states
declare -A WORKER_JOB_IDS
declare -A WORKER_RETRIES
declare -a EXCLUDED_NODES=()

# Function to submit a worker job
submit_worker() {
    local worker_id=$1
    local start_idx=$((PATIENT_OFFSET + worker_id * PATIENTS_PER_WORKER))
    local end_idx=$((PATIENT_OFFSET + (worker_id + 1) * PATIENTS_PER_WORKER))

    # Handle last worker
    if [ $end_idx -gt $((PATIENT_OFFSET + TOTAL_PATIENTS)) ]; then
        end_idx=$((PATIENT_OFFSET + TOTAL_PATIENTS))
    fi

    if [ $start_idx -ge $((PATIENT_OFFSET + TOTAL_PATIENTS)) ]; then
        echo "[sft master] Worker ${worker_id} has no patients to process, skipping"
        return 1
    fi

    local exclude_arg=""
    if [ ${#EXCLUDED_NODES[@]} -gt 0 ]; then
        exclude_arg="--exclude=$(IFS=,; echo "${EXCLUDED_NODES[*]}")"
    fi

    local gpu_spec="h200:${NUM_GPUS}"

    echo "[sft master] Submitting worker ${worker_id} for patients [${start_idx}:${end_idx}]..."

    local job_id
    job_id=$(sbatch --parsable ${exclude_arg} \
        --gres=gpu:${gpu_spec} \
        --cpus-per-task="${WORKER_CPUS}" \
        --mem="${WORKER_MEM}" \
        --time="${WORKER_TIME}" \
        --export=ALL,MODEL_ID="${MODEL_ID}",BASE_MODEL_PATH="${BASE_MODEL_PATH}",LORA_PATH="${LORA_PATH}",PATIENT_START="${start_idx}",PATIENT_END="${end_idx}",WORKER_ID="${worker_id}",VERSION="${VERSION}",WORKER_OUTPUT_DIR="${WORKER_OUTPUT_DIR}",SIGNAL_DIR="${SIGNAL_DIR}",CONDA_ENV="${CONDA_ENV}",NUM_GPUS="${NUM_GPUS}",EXTRACTOR_MODEL="${EXTRACTOR_MODEL:-}",MAX_TOKENS="${MAX_TOKENS}" \
        "${WORKER_SCRIPT}")

    WORKER_JOB_IDS[$worker_id]=$job_id
    echo "[sft master] Worker ${worker_id} submitted with job ID: ${job_id}"
    return 0
}

# Function to check if worker completed
is_worker_complete() {
    local worker_id=$1
    local signal_file="${SIGNAL_DIR}/${MODEL_ID}_sft_vllm_worker${worker_id}.done"
    local json_output="${WORKER_OUTPUT_DIR}/${MODEL_ID}_sft_vllm_worker${worker_id}_v${VERSION}.json"

    if [ -f "$signal_file" ]; then
        return 0
    fi

    if [ -f "$json_output" ]; then
        if "${CONDA_PREFIX}/bin/python" -c "import json; d=json.load(open('${json_output}')); assert len(d.get('results',[])) > 0" 2>/dev/null; then
            touch "$signal_file" 2>/dev/null || true
            return 0
        fi
    fi

    return 1
}

# Function to get job state
get_job_state() {
    local job_id=$1
    sacct -j "$job_id" --format=State --parsable --noheader 2>/dev/null | head -1 | cut -d'|' -f1
}

# Function to get failed node
get_failed_node() {
    local job_id=$1
    sacct -j "$job_id" --format=NodeList --parsable --noheader 2>/dev/null | head -1 | cut -d'|' -f1
}

# Submit all workers initially
echo ""
echo "[sft master] Submitting ${NUM_WORKERS} worker jobs..."
ACTUAL_WORKERS=0
for ((i=0; i<NUM_WORKERS; i++)); do
    WORKER_RETRIES[$i]=0
    if submit_worker $i; then
        ((++ACTUAL_WORKERS))
    fi
    sleep 1
done

if [ $ACTUAL_WORKERS -eq 0 ]; then
    echo "[sft master] ERROR: No workers submitted"
    exit 1
fi

echo ""
echo "[sft master] ${ACTUAL_WORKERS} workers submitted. Starting monitoring loop..."
echo "[sft master] Poll interval: ${POLL_INTERVAL} seconds"
echo ""

# Monitoring loop
while true; do
    completed_count=0
    pending_count=0
    running_count=0
    failed_workers=()

    for ((i=0; i<ACTUAL_WORKERS; i++)); do
        if is_worker_complete $i; then
            ((++completed_count))
        else
            job_id="${WORKER_JOB_IDS[$i]:-}"
            if [ -z "$job_id" ]; then
                continue
            fi

            state=$(get_job_state "$job_id")

            case "$state" in
                COMPLETED)
                    if is_worker_complete $i; then
                        ((++completed_count))
                    else
                        echo "[sft master] Warning: Worker ${i} marked COMPLETED but no valid output"
                        failed_workers+=($i)
                    fi
                    ;;
                PENDING)
                    ((++pending_count))
                    ;;
                RUNNING)
                    ((++running_count))
                    ;;
                FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|CANCELLED)
                    failed_workers+=($i)
                    ;;
                "")
                    ((++pending_count))
                    ;;
                *)
                    ((++pending_count))
                    ;;
            esac
        fi
    done

    echo "[sft master] Status: ${completed_count}/${ACTUAL_WORKERS} complete, ${running_count} running, ${pending_count} pending, ${#failed_workers[@]} failed"

    # Handle failed workers
    for worker_id in "${failed_workers[@]}"; do
        retry_count=${WORKER_RETRIES[$worker_id]}
        if [ $retry_count -lt $MAX_RETRIES ]; then
            failed_job_id="${WORKER_JOB_IDS[$worker_id]}"
            failed_node=$(get_failed_node "$failed_job_id")

            echo "[sft master] Worker ${worker_id} failed (attempt $((retry_count + 1))/${MAX_RETRIES})"

            if [ -n "$failed_node" ] && [[ ! " ${EXCLUDED_NODES[*]} " =~ " ${failed_node} " ]]; then
                EXCLUDED_NODES+=("$failed_node")
                echo "[sft master] Adding node ${failed_node} to exclusion list"
            fi

            WORKER_RETRIES[$worker_id]=$((retry_count + 1))
            submit_worker $worker_id || true
        else
            echo "[sft master] Worker ${worker_id} exhausted all retries (${MAX_RETRIES})"
        fi
    done

    # Check if all done
    if [ $completed_count -eq $ACTUAL_WORKERS ]; then
        echo ""
        echo "[sft master] All workers completed successfully!"
        break
    fi

    # Check for permanent failures
    permanent_failures=0
    for ((i=0; i<ACTUAL_WORKERS; i++)); do
        if ! is_worker_complete $i && [ ${WORKER_RETRIES[$i]} -ge $MAX_RETRIES ]; then
            job_state=$(get_job_state "${WORKER_JOB_IDS[$i]:-}")
            if [[ "$job_state" =~ ^(FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|CANCELLED)$ ]]; then
                ((++permanent_failures))
            fi
        fi
    done

    if [ $permanent_failures -gt 0 ] && [ $((completed_count + running_count + pending_count)) -eq $completed_count ]; then
        echo ""
        echo "[sft master] ERROR: ${permanent_failures} workers permanently failed."
        echo "[sft master] Proceeding with available results..."
        break
    fi

    sleep "${POLL_INTERVAL}"
done

# Merge results
echo ""
echo "[sft master] Merging worker results..."

FINAL_OUTPUT="${OUTPUT_BASE_DIR}/${MODEL_ID}_sft_${TOTAL_PATIENTS}-v${VERSION}.json"

${CONDA_PREFIX}/bin/python "${MERGE_SCRIPT}" \
    --input-dir "${WORKER_OUTPUT_DIR}" \
    --output "${FINAL_OUTPUT}" \
    --model-id "${MODEL_ID}" \
    --version "${VERSION}"

if [ -f "${FINAL_OUTPUT}" ]; then
    patient_count=$(${CONDA_PREFIX}/bin/python -c "import json; print(len(json.load(open('${FINAL_OUTPUT}'))['results']))")
    echo "[sft master] Merged output saved to: ${FINAL_OUTPUT}"
    echo "[sft master] Total patients in merged output: ${patient_count}"
else
    echo "[sft master] ERROR: Merge failed. Check individual worker outputs."
    exit 1
fi

echo ""
echo "============================================================"
echo "EvidenceRL SFT Generation Complete (vLLM)"
echo "============================================================"
echo "Mode:      SFT LoRA (no-RAG)"
echo "Inference: vLLM + LoRA"
echo "Model:     ${MODEL_ID}"
echo "Output:    ${FINAL_OUTPUT}"
echo "============================================================"
