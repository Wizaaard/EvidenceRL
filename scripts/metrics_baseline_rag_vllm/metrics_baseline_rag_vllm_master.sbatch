#!/usr/bin/env bash
#SBATCH -J erl_baseline_rag_metrics_vllm_master
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH -o logs_evidencerl_baseline_rag/metrics_baseline_rag_vllm_master_%j.out
#SBATCH -e logs_evidencerl_baseline_rag/metrics_baseline_rag_vllm_master_%j.err

# ============================================================================
# EvidenceRL Baseline RAG Metrics Master (vLLM Version)
# ============================================================================

set -euo pipefail

: "${MODEL_ID:?Need MODEL_ID}"
: "${MODEL_NAME:?Need MODEL_NAME}"
: "${VERSION:=1.0}"
: "${GENERATION_FILE:?Need GENERATION_FILE}"
: "${JUDGE_MODEL:?Need JUDGE_MODEL}"

NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_RETRIES="${MAX_RETRIES:-3}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

PROJECT_ROOT="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code"
METRICS_OUTPUT_DIR="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/metrics_baseline_rag_output"
WORKER_SCRIPT="${PROJECT_ROOT}/scripts/metrics_baseline_rag_vllm/metrics_baseline_rag_vllm_worker.sbatch"
MERGE_SCRIPT="${PROJECT_ROOT}/scripts/metrics_vllm/merge_metrics_results_vllm.py"

SIGNAL_DIR="${METRICS_OUTPUT_DIR}/${MODEL_ID}_v${VERSION}/signals"
WORKER_OUTPUT_DIR="${METRICS_OUTPUT_DIR}/${MODEL_ID}_v${VERSION}/workers"
FINAL_OUTPUT_DIR="${METRICS_OUTPUT_DIR}/${MODEL_ID}_v${VERSION}"
LOG_DIR="${PROJECT_ROOT}/logs_evidencerl_baseline_rag"

mkdir -p "${SIGNAL_DIR}" "${WORKER_OUTPUT_DIR}" "${FINAL_OUTPUT_DIR}" "${LOG_DIR}"

echo "Reading baseline RAG generation file: ${GENERATION_FILE}"
TOTAL_PATIENTS=$(python3 -c "import json; print(len(json.load(open('${GENERATION_FILE}'))['results']))")
echo "Found ${TOTAL_PATIENTS} patients"

PATIENTS_PER_WORKER=$(( (TOTAL_PATIENTS + NUM_WORKERS - 1) / NUM_WORKERS ))

echo "============================================================"
echo "EvidenceRL Baseline RAG Metrics Master (vLLM Version)"
echo "============================================================"
echo "MODE:            BASELINE RAG (with retrieval)"
echo "INFERENCE:       vLLM"
echo "MODEL_ID:        ${MODEL_ID}"
echo "GENERATION_FILE: ${GENERATION_FILE}"
echo "JUDGE_MODEL:     ${JUDGE_MODEL}"
echo "TOTAL_PATIENTS:  ${TOTAL_PATIENTS}"
echo "NUM_WORKERS:     ${NUM_WORKERS}"
echo "============================================================"

# Initialize conda
for conda_path in "$HOME/.conda/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh" "/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh"; do
    if [ -f "$conda_path" ]; then
        source "$conda_path"
        break
    fi
done

CONDA_ENV="${CONDA_ENV:-evidencerl}"
set +u; conda activate "${CONDA_ENV}" 2>/dev/null || true; set -u

declare -A WORKER_JOB_IDS
declare -A WORKER_RETRIES
declare -a EXCLUDED_NODES=()

submit_worker() {
    local worker_id=$1
    local start_idx=$((worker_id * PATIENTS_PER_WORKER))
    local end_idx=$(( (worker_id + 1) * PATIENTS_PER_WORKER ))
    [ $end_idx -gt $TOTAL_PATIENTS ] && end_idx=$TOTAL_PATIENTS
    [ $start_idx -ge $TOTAL_PATIENTS ] && return 1

    local exclude_arg=""
    [ ${#EXCLUDED_NODES[@]} -gt 0 ] && exclude_arg="--exclude=$(IFS=,; echo "${EXCLUDED_NODES[*]}")"

    echo "[master] Submitting vLLM metrics worker ${worker_id} for patients [${start_idx}:${end_idx}]..."

    local job_id
    job_id=$(sbatch --parsable ${exclude_arg} \
        --export=ALL,MODEL_ID="${MODEL_ID}",GENERATION_FILE="${GENERATION_FILE}",JUDGE_MODEL="${JUDGE_MODEL}",PATIENT_START="${start_idx}",PATIENT_END="${end_idx}",WORKER_ID="${worker_id}",VERSION="${VERSION}",OUTPUT_BASE_DIR="${WORKER_OUTPUT_DIR}",SIGNAL_DIR="${SIGNAL_DIR}",CONDA_ENV="${CONDA_ENV}",TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}",GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
        "${WORKER_SCRIPT}")

    WORKER_JOB_IDS[$worker_id]=$job_id
    echo "[master] Worker ${worker_id} submitted: ${job_id}"
    return 0
}

is_worker_complete() {
    local worker_id=$1
    local signal_file="${SIGNAL_DIR}/${MODEL_ID}_baseline_rag_metrics_vllm_worker${worker_id}.done"
    local output_file="${WORKER_OUTPUT_DIR}/${MODEL_ID}_baseline_rag_metrics_vllm_worker${worker_id}_v${VERSION}.json"
    [ -f "$signal_file" ] && return 0
    if [ -f "$output_file" ]; then
        python3 -c "import json; d=json.load(open('${output_file}')); assert len(d.get('patient_metrics',[])) > 0" 2>/dev/null && { touch "$signal_file"; return 0; }
    fi
    return 1
}

get_job_state() { sacct -j "$1" --format=State --parsable --noheader 2>/dev/null | head -1 | cut -d'|' -f1; }
get_failed_node() { sacct -j "$1" --format=NodeList --parsable --noheader 2>/dev/null | head -1 | cut -d'|' -f1; }

echo "[master] Submitting ${NUM_WORKERS} vLLM metrics workers..."
ACTUAL_WORKERS=0
for ((i=0; i<NUM_WORKERS; i++)); do
    WORKER_RETRIES[$i]=0
    submit_worker $i && ((++ACTUAL_WORKERS))
    sleep 1
done

[ $ACTUAL_WORKERS -eq 0 ] && { echo "[master] ERROR: No workers submitted"; exit 1; }

echo "[master] ${ACTUAL_WORKERS} workers submitted. Monitoring..."

while true; do
    completed_count=0; pending_count=0; running_count=0; failed_workers=()

    for ((i=0; i<ACTUAL_WORKERS; i++)); do
        if is_worker_complete $i; then
            ((++completed_count))
        else
            job_id="${WORKER_JOB_IDS[$i]:-}"
            [ -z "$job_id" ] && continue
            state=$(get_job_state "$job_id")
            case "$state" in
                COMPLETED) is_worker_complete $i && ((++completed_count)) || failed_workers+=($i) ;;
                PENDING) ((++pending_count)) ;;
                RUNNING) ((++running_count)) ;;
                FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|CANCELLED) failed_workers+=($i) ;;
                *) ((++pending_count)) ;;
            esac
        fi
    done

    echo "[master] Status: ${completed_count}/${ACTUAL_WORKERS} complete, ${running_count} running, ${pending_count} pending"

    for worker_id in "${failed_workers[@]}"; do
        retry_count=${WORKER_RETRIES[$worker_id]}
        if [ $retry_count -lt $MAX_RETRIES ]; then
            failed_node=$(get_failed_node "${WORKER_JOB_IDS[$worker_id]}")
            [ -n "$failed_node" ] && EXCLUDED_NODES+=("$failed_node")
            WORKER_RETRIES[$worker_id]=$((retry_count + 1))
            submit_worker $worker_id || true
        fi
    done

    [ $completed_count -eq $ACTUAL_WORKERS ] && { echo "[master] All workers completed!"; break; }
    sleep "${POLL_INTERVAL}"
done

echo "[master] Merging results..."
FINAL_OUTPUT="${FINAL_OUTPUT_DIR}/${MODEL_ID}_baseline_rag_metrics-v${VERSION}.json"

python "${MERGE_SCRIPT}" \
    --input-dir "${WORKER_OUTPUT_DIR}" \
    --output "${FINAL_OUTPUT}" \
    --num-workers "${ACTUAL_WORKERS}" \
    --model-id "${MODEL_ID}" \
    --version "${VERSION}" \
    --metrics-type "baseline_rag"

if [ -f "${FINAL_OUTPUT}" ]; then
    patient_count=$(python -c "import json; print(len(json.load(open('${FINAL_OUTPUT}'))['patient_metrics']))")
    echo "[master] Saved: ${FINAL_OUTPUT} (${patient_count} patients)"
else
    echo "[master] ERROR: Merge failed"
    exit 1
fi

echo "============================================================"
echo "Baseline RAG Metrics Complete (vLLM): ${FINAL_OUTPUT}"
echo "============================================================"
