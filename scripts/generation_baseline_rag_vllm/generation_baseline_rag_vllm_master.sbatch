#!/usr/bin/env bash
#SBATCH -J erl_baseline_rag_vllm_master
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -o logs_evidencerl_baseline_rag/generation_baseline_rag_vllm_master_%j.out
#SBATCH -e logs_evidencerl_baseline_rag/generation_baseline_rag_vllm_master_%j.err

# ============================================================================
# EvidenceRL Baseline RAG Generation Master (vLLM Version)
# ============================================================================
# Orchestrates distributed BASELINE RAG generation (WITH retrieval) using vLLM
# on held-out test patients (1000-1099).
#
# Required environment variables:
#   MODEL_ID        - Short identifier for the model
#   MODEL_NAME      - Full path to the model
#   VERSION         - Version string for output naming
#   FAISS_INDEX_DIR - Path to pre-built FAISS index
# ============================================================================

set -euo pipefail

# Validate required environment variables
: "${MODEL_ID:?Need MODEL_ID}"
: "${MODEL_NAME:?Need MODEL_NAME}"
: "${VERSION:=1.0}"
: "${FAISS_INDEX_DIR:?Need FAISS_INDEX_DIR}"

# Configuration - 1 worker per 100 patients
PATIENT_OFFSET="${PATIENT_OFFSET:-1000}"
TOTAL_PATIENTS="${TOTAL_PATIENTS:-100}"
PATIENTS_PER_WORKER=100
NUM_WORKERS=$(( (TOTAL_PATIENTS + PATIENTS_PER_WORKER - 1) / PATIENTS_PER_WORKER ))
NUM_GPUS="${NUM_GPUS:-2}"
WORKER_CPUS="${WORKER_CPUS:-16}"
WORKER_MEM="${WORKER_MEM:-128G}"
WORKER_TIME="${WORKER_TIME:-02:00:00}"
MAX_RETRIES="${MAX_RETRIES:-3}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"

# Paths - BASELINE RAG output directory
PROJECT_ROOT="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code"
OUTPUT_BASE_DIR="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/generation_baseline_rag_output"
WORKER_SCRIPT="${PROJECT_ROOT}/scripts/generation_baseline_rag_vllm/generation_baseline_rag_vllm_worker.sbatch"
MERGE_SCRIPT="${PROJECT_ROOT}/scripts/generation/merge_generation_results.py"

# Create directories
SIGNAL_DIR="${OUTPUT_BASE_DIR}/${MODEL_ID}_v${VERSION}/signals"
WORKER_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_ID}_v${VERSION}/workers"
FINAL_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_ID}_v${VERSION}"
LOG_DIR="${PROJECT_ROOT}/logs_evidencerl_baseline_rag"

mkdir -p "${SIGNAL_DIR}" "${WORKER_OUTPUT_DIR}" "${FINAL_OUTPUT_DIR}" "${LOG_DIR}"

echo "============================================================"
echo "EvidenceRL Baseline RAG Generation Master (vLLM Version)"
echo "============================================================"
echo "MODE:            BASELINE RAG (with retrieval)"
echo "INFERENCE:       vLLM"
echo "MODEL_ID:        ${MODEL_ID}"
echo "MODEL_NAME:      ${MODEL_NAME}"
echo "FAISS_INDEX:     ${FAISS_INDEX_DIR}"
echo "VERSION:         ${VERSION}"
echo "PATIENT_RANGE:   ${PATIENT_OFFSET} to $((PATIENT_OFFSET + TOTAL_PATIENTS)) (held-out test set)"
echo "TOTAL_PATIENTS:  ${TOTAL_PATIENTS}"
echo "NUM_WORKERS:     ${NUM_WORKERS} (1 worker per 100 patients)"
echo "PATIENTS/WORKER: ${PATIENTS_PER_WORKER}"
echo "WORKER_RESOURCES: ${NUM_GPUS}x GPU, ${WORKER_CPUS} CPUs, ${WORKER_MEM}, ${WORKER_TIME}"
echo "OUTPUT_DIR:      ${FINAL_OUTPUT_DIR}"
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
    if [ $worker_id -eq $((NUM_WORKERS - 1)) ]; then
        end_idx=$((PATIENT_OFFSET + TOTAL_PATIENTS))
    fi

    # Build exclude node list
    local exclude_arg=""
    if [ ${#EXCLUDED_NODES[@]} -gt 0 ]; then
        exclude_arg="--exclude=$(IFS=,; echo "${EXCLUDED_NODES[*]}")"
    fi

    local resource_args="--gres=gpu:h200:${NUM_GPUS} --mem=${WORKER_MEM} --cpus-per-task=${WORKER_CPUS} --time=${WORKER_TIME}"

    echo "[master] Submitting vLLM baseline RAG worker ${worker_id} for patients [${start_idx}:${end_idx}]..."

    local job_id
    job_id=$(sbatch --parsable ${exclude_arg} ${resource_args} \
        --export=ALL,MODEL_ID="${MODEL_ID}",MODEL_NAME="${MODEL_NAME}",PATIENT_START="${start_idx}",PATIENT_END="${end_idx}",WORKER_ID="${worker_id}",VERSION="${VERSION}",OUTPUT_BASE_DIR="${WORKER_OUTPUT_DIR}",SIGNAL_DIR="${SIGNAL_DIR}",CONDA_ENV="${CONDA_ENV}",FAISS_INDEX_DIR="${FAISS_INDEX_DIR}",NUM_GPUS="${NUM_GPUS}" \
        "${WORKER_SCRIPT}")

    WORKER_JOB_IDS[$worker_id]=$job_id
    echo "[master] Worker ${worker_id} submitted with job ID: ${job_id}"
}

# Function to check if worker completed
is_worker_complete() {
    local worker_id=$1
    local signal_file="${SIGNAL_DIR}/${MODEL_ID}_baseline_rag_vllm_worker${worker_id}.done"
    local output_file="${WORKER_OUTPUT_DIR}/${MODEL_ID}_baseline_rag_vllm_worker${worker_id}_v${VERSION}.json"

    if [ -f "$signal_file" ]; then
        return 0
    fi

    if [ -f "$output_file" ]; then
        if python3 -c "import json; d=json.load(open('${output_file}')); assert len(d.get('results',[])) > 0" 2>/dev/null; then
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

# Submit all workers
echo ""
echo "[master] Submitting ${NUM_WORKERS} vLLM baseline RAG worker jobs..."
for ((i=0; i<NUM_WORKERS; i++)); do
    WORKER_RETRIES[$i]=0
    submit_worker $i
    sleep 1
done

echo ""
echo "[master] All workers submitted. Starting monitoring loop..."
echo ""

# Monitoring loop
while true; do
    completed_count=0
    pending_count=0
    running_count=0
    failed_workers=()

    for ((i=0; i<NUM_WORKERS; i++)); do
        if is_worker_complete $i; then
            ((++completed_count))
        else
            job_id="${WORKER_JOB_IDS[$i]}"
            state=$(get_job_state "$job_id")

            case "$state" in
                COMPLETED)
                    if is_worker_complete $i; then
                        ((++completed_count))
                    else
                        failed_workers+=($i)
                    fi
                    ;;
                PENDING) ((++pending_count)) ;;
                RUNNING) ((++running_count)) ;;
                FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|CANCELLED) failed_workers+=($i) ;;
                "") ((++pending_count)) ;;
                *) ((++pending_count)) ;;
            esac
        fi
    done

    echo "[master] Status: ${completed_count}/${NUM_WORKERS} complete, ${running_count} running, ${pending_count} pending, ${#failed_workers[@]} failed"

    # Handle failed workers
    for worker_id in "${failed_workers[@]}"; do
        retry_count=${WORKER_RETRIES[$worker_id]}
        if [ $retry_count -lt $MAX_RETRIES ]; then
            failed_job_id="${WORKER_JOB_IDS[$worker_id]}"
            failed_node=$(get_failed_node "$failed_job_id")

            echo "[master] Worker ${worker_id} failed (attempt $((retry_count + 1))/${MAX_RETRIES})"

            if [ -n "$failed_node" ] && [[ ! " ${EXCLUDED_NODES[*]} " =~ " ${failed_node} " ]]; then
                EXCLUDED_NODES+=("$failed_node")
            fi

            WORKER_RETRIES[$worker_id]=$((retry_count + 1))
            submit_worker $worker_id
        fi
    done

    if [ $completed_count -eq $NUM_WORKERS ]; then
        echo ""
        echo "[master] All workers completed successfully!"
        break
    fi

    # Check for permanent failures
    permanent_failures=0
    for ((i=0; i<NUM_WORKERS; i++)); do
        if ! is_worker_complete $i && [ ${WORKER_RETRIES[$i]} -ge $MAX_RETRIES ]; then
            job_state=$(get_job_state "${WORKER_JOB_IDS[$i]}")
            if [[ "$job_state" =~ ^(FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|CANCELLED)$ ]]; then
                ((++permanent_failures))
            fi
        fi
    done

    if [ $permanent_failures -gt 0 ] && [ $((completed_count + running_count + pending_count)) -eq $completed_count ]; then
        echo "[master] ERROR: ${permanent_failures} workers permanently failed."
        break
    fi

    sleep "${POLL_INTERVAL}"
done

# Merge results
echo ""
echo "[master] Merging worker results..."

FINAL_OUTPUT="${FINAL_OUTPUT_DIR}/${MODEL_ID}_baseline_rag_${TOTAL_PATIENTS}-v${VERSION}.json"

python "${MERGE_SCRIPT}" \
    --input-dir "${WORKER_OUTPUT_DIR}" \
    --output "${FINAL_OUTPUT}" \
    --model-id "${MODEL_ID}" \
    --version "${VERSION}"

if [ -f "${FINAL_OUTPUT}" ]; then
    echo "[master] Merged results saved to: ${FINAL_OUTPUT}"
    result_count=$(python -c "import json; print(len(json.load(open('${FINAL_OUTPUT}'))['results']))")
    echo "[master] Total patients in merged output: ${result_count}"
else
    echo "[master] ERROR: Merge failed."
    exit 1
fi

echo ""
echo "============================================================"
echo "EvidenceRL Baseline RAG Generation Complete (vLLM Version)"
echo "============================================================"
echo "Mode:      BASELINE RAG (with retrieval)"
echo "Inference: vLLM"
echo "Model:     ${MODEL_ID}"
echo "Output:    ${FINAL_OUTPUT}"
echo "============================================================"
