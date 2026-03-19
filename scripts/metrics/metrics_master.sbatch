#!/usr/bin/env bash
#SBATCH -J erl_metrics_master
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -o logs_evidencerl/metrics_master_%j.out
#SBATCH -e logs_evidencerl/metrics_master_%j.err

# ============================================================================
# EvidenceRL Metrics Computation Master
# ============================================================================
# Orchestrates distributed metrics computation by:
# 1. Reading generation output to determine patient count
# 2. Submitting worker jobs (each processes a subset of patients)
# 3. Monitoring worker completion via signal files
# 4. Retrying failed workers with node exclusion
# 5. Merging results when all workers complete
#
# Required environment variables:
#   MODEL_ID        - Short model identifier
#   MODEL_NAME      - Full model name
#   VERSION         - Version string
#   GENERATION_FILE - Path to generation JSON output
#   JUDGE_MODEL     - Path to judge model (fixed: medgemma-27b-it)
#
# Optional environment variables:
#   NUM_WORKERS     - Number of worker jobs (default: 10)
#   MAX_RETRIES     - Max retries per worker (default: 3)
#   POLL_INTERVAL   - Seconds between status checks (default: 60)
# ============================================================================

set -euo pipefail

# Validate required environment variables
: "${MODEL_ID:?Need MODEL_ID}"
: "${MODEL_NAME:?Need MODEL_NAME}"
: "${VERSION:=1.0}"
: "${GENERATION_FILE:?Need GENERATION_FILE}"
: "${JUDGE_MODEL:?Need JUDGE_MODEL}"

# Configuration
NUM_WORKERS="${NUM_WORKERS:-10}"
MAX_RETRIES="${MAX_RETRIES:-3}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"

# Paths
PROJECT_ROOT="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code"
METRICS_OUTPUT_DIR="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/metrics_output"
WORKER_SCRIPT="${PROJECT_ROOT}/scripts/metrics/metrics_worker.sbatch"
MERGE_SCRIPT="${PROJECT_ROOT}/scripts/metrics/merge_metrics_results.py"

# Create directories
SIGNAL_DIR="${METRICS_OUTPUT_DIR}/${MODEL_ID}_v${VERSION}/signals"
WORKER_OUTPUT_DIR="${METRICS_OUTPUT_DIR}/${MODEL_ID}_v${VERSION}/workers"
FINAL_OUTPUT_DIR="${METRICS_OUTPUT_DIR}/${MODEL_ID}_v${VERSION}"
LOG_DIR="${PROJECT_ROOT}/logs_evidencerl"

mkdir -p "${SIGNAL_DIR}" "${WORKER_OUTPUT_DIR}" "${FINAL_OUTPUT_DIR}" "${LOG_DIR}"

# Get patient count from generation file
echo "Reading generation file: ${GENERATION_FILE}"
TOTAL_PATIENTS=$(python3 -c "import json; print(len(json.load(open('${GENERATION_FILE}'))['results']))")
echo "Found ${TOTAL_PATIENTS} patients in generation output"

# Calculate patients per worker
PATIENTS_PER_WORKER=$(( (TOTAL_PATIENTS + NUM_WORKERS - 1) / NUM_WORKERS ))

echo "============================================================"
echo "EvidenceRL Metrics Master"
echo "============================================================"
echo "MODEL_ID:        ${MODEL_ID}"
echo "MODEL_NAME:      ${MODEL_NAME}"
echo "VERSION:         ${VERSION}"
echo "GENERATION_FILE: ${GENERATION_FILE}"
echo "JUDGE_MODEL:     ${JUDGE_MODEL} (fixed: medgemma-27b-it)"
echo "TOTAL_PATIENTS:  ${TOTAL_PATIENTS}"
echo "NUM_WORKERS:     ${NUM_WORKERS}"
echo "PATIENTS/WORKER: ~${PATIENTS_PER_WORKER}"
echo "OUTPUT_DIR:      ${FINAL_OUTPUT_DIR}"
echo "============================================================"

# Initialize conda for running merge script later
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
    local start_idx=$((worker_id * PATIENTS_PER_WORKER))
    local end_idx=$(( (worker_id + 1) * PATIENTS_PER_WORKER ))

    # Clamp to total patients
    if [ $end_idx -gt $TOTAL_PATIENTS ]; then
        end_idx=$TOTAL_PATIENTS
    fi

    # Skip if no patients to process
    if [ $start_idx -ge $TOTAL_PATIENTS ]; then
        echo "[master] Worker ${worker_id} has no patients to process, skipping"
        return 1
    fi

    # Build exclude node list
    local exclude_arg=""
    if [ ${#EXCLUDED_NODES[@]} -gt 0 ]; then
        exclude_arg="--exclude=$(IFS=,; echo "${EXCLUDED_NODES[*]}")"
    fi

    echo "[master] Submitting worker ${worker_id} for patients [${start_idx}:${end_idx}]..."

    local job_id
    job_id=$(sbatch --parsable ${exclude_arg} \
        --export=ALL,MODEL_ID="${MODEL_ID}",GENERATION_FILE="${GENERATION_FILE}",JUDGE_MODEL="${JUDGE_MODEL}",PATIENT_START="${start_idx}",PATIENT_END="${end_idx}",WORKER_ID="${worker_id}",VERSION="${VERSION}",OUTPUT_BASE_DIR="${WORKER_OUTPUT_DIR}",SIGNAL_DIR="${SIGNAL_DIR}",CONDA_ENV="${CONDA_ENV}" \
        "${WORKER_SCRIPT}")

    WORKER_JOB_IDS[$worker_id]=$job_id
    echo "[master] Worker ${worker_id} submitted with job ID: ${job_id}"
    return 0
}

# Function to check if worker completed
is_worker_complete() {
    local worker_id=$1
    local signal_file="${SIGNAL_DIR}/${MODEL_ID}_metrics_worker${worker_id}.done"
    local output_file="${WORKER_OUTPUT_DIR}/${MODEL_ID}_metrics_worker${worker_id}_v${VERSION}.json"

    if [ -f "$signal_file" ]; then
        return 0
    fi

    if [ -f "$output_file" ]; then
        if python3 -c "import json; d=json.load(open('${output_file}')); assert len(d.get('patient_metrics',[])) > 0" 2>/dev/null; then
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
echo "[master] Submitting ${NUM_WORKERS} worker jobs..."
ACTUAL_WORKERS=0
for ((i=0; i<NUM_WORKERS; i++)); do
    WORKER_RETRIES[$i]=0
    if submit_worker $i; then
        ((++ACTUAL_WORKERS))
    fi
    sleep 1
done

if [ $ACTUAL_WORKERS -eq 0 ]; then
    echo "[master] ERROR: No workers submitted"
    exit 1
fi

echo ""
echo "[master] ${ACTUAL_WORKERS} workers submitted. Starting monitoring loop..."
echo "[master] Poll interval: ${POLL_INTERVAL} seconds"
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
                        echo "[master] Warning: Worker ${i} marked COMPLETED but no valid output"
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

    echo "[master] Status: ${completed_count}/${ACTUAL_WORKERS} complete, ${running_count} running, ${pending_count} pending, ${#failed_workers[@]} failed"

    # Handle failed workers
    for worker_id in "${failed_workers[@]}"; do
        retry_count=${WORKER_RETRIES[$worker_id]}
        if [ $retry_count -lt $MAX_RETRIES ]; then
            failed_job_id="${WORKER_JOB_IDS[$worker_id]}"
            failed_node=$(get_failed_node "$failed_job_id")

            echo "[master] Worker ${worker_id} failed (attempt $((retry_count + 1))/${MAX_RETRIES})"

            if [ -n "$failed_node" ] && [[ ! " ${EXCLUDED_NODES[*]} " =~ " ${failed_node} " ]]; then
                EXCLUDED_NODES+=("$failed_node")
                echo "[master] Adding node ${failed_node} to exclusion list"
            fi

            WORKER_RETRIES[$worker_id]=$((retry_count + 1))
            submit_worker $worker_id || true
        else
            echo "[master] Worker ${worker_id} exhausted all retries (${MAX_RETRIES})"
        fi
    done

    # Check if all done
    if [ $completed_count -eq $ACTUAL_WORKERS ]; then
        echo ""
        echo "[master] All workers completed successfully!"
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
        echo "[master] ERROR: ${permanent_failures} workers permanently failed."
        echo "[master] Proceeding with available results..."
        break
    fi

    sleep "${POLL_INTERVAL}"
done

# Merge results
echo ""
echo "[master] Merging worker results..."

FINAL_OUTPUT="${FINAL_OUTPUT_DIR}/${MODEL_ID}_metrics-v${VERSION}.json"

python "${MERGE_SCRIPT}" \
    --input-dir "${WORKER_OUTPUT_DIR}" \
    --output "${FINAL_OUTPUT}" \
    --model-id "${MODEL_ID}" \
    --version "${VERSION}"

if [ -f "${FINAL_OUTPUT}" ]; then
    echo "[master] Merged metrics saved to: ${FINAL_OUTPUT}"

    patient_count=$(python -c "import json; print(len(json.load(open('${FINAL_OUTPUT}'))['patient_metrics']))")
    echo "[master] Total patients in merged output: ${patient_count}"
else
    echo "[master] ERROR: Merge failed. Check individual worker outputs in: ${WORKER_OUTPUT_DIR}"
    exit 1
fi

echo ""
echo "============================================================"
echo "EvidenceRL Metrics Computation Complete"
echo "============================================================"
echo "Model:   ${MODEL_ID}"
echo "Output:  ${FINAL_OUTPUT}"
echo "Workers: ${WORKER_OUTPUT_DIR}/"
echo "============================================================"
