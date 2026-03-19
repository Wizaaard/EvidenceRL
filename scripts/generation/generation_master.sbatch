#!/usr/bin/env bash
#SBATCH -J erl_master
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -o logs_evidencerl/master_%j.out
#SBATCH -e logs_evidencerl/master_%j.err

# ============================================================================
# EvidenceRL Generation Master
# ============================================================================
# This script orchestrates distributed generation by:
# 1. Submitting worker jobs with configurable resources
# 2. Monitoring worker completion via signal files
# 3. Retrying failed workers with node exclusion
# 4. Merging results when all workers complete
#
# Resource allocation per worker (configurable via env vars):
#   Standard models (NUM_GPUS=2):    2x H200, 128GB RAM, 16 CPUs, 4h
#   Large models (NUM_GPUS=4):       4x H200, 256GB RAM, 32 CPUs, 4h
#   Extra-large models (NUM_GPUS=8): 8x H200, 256GB RAM, 64 CPUs, 2h
#
# Required environment variables:
#   MODEL_ID        - Short identifier for the model
#   MODEL_NAME      - Full path to the model
#   VERSION         - Version string for output naming
#
# Optional environment variables:
#   TOTAL_PATIENTS  - Total number of patients (default: 1000)
#   NUM_WORKERS     - Number of worker jobs (default: 20)
#   NUM_GPUS        - GPUs per worker (default: 2)
#   WORKER_CPUS     - CPUs per worker (default: 16)
#   WORKER_MEM      - Memory per worker (default: 128G)
#   WORKER_TIME     - Time limit per worker (default: 04:00:00)
#   MAX_RETRIES     - Max retries per worker (default: 3)
#   POLL_INTERVAL   - Seconds between status checks (default: 60)
# ============================================================================

set -euo pipefail

# Validate required environment variables
: "${MODEL_ID:?Need MODEL_ID}"
: "${MODEL_NAME:?Need MODEL_NAME}"
: "${VERSION:=1.0}"

# Configuration
TOTAL_PATIENTS="${TOTAL_PATIENTS:-1000}"
NUM_WORKERS="${NUM_WORKERS:-20}"
NUM_GPUS="${NUM_GPUS:-2}"
WORKER_CPUS="${WORKER_CPUS:-16}"
WORKER_MEM="${WORKER_MEM:-128G}"
WORKER_TIME="${WORKER_TIME:-04:00:00}"
MAX_RETRIES="${MAX_RETRIES:-3}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"

# Paths
PROJECT_ROOT="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code"
OUTPUT_BASE_DIR="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/generation_output"
WORKER_SCRIPT="${PROJECT_ROOT}/scripts/generation/generation_worker.sbatch"
MERGE_SCRIPT="${PROJECT_ROOT}/scripts/generation/merge_generation_results.py"

# Pre-built FAISS index (significantly speeds up worker startup)
FAISS_INDEX_DIR="${FAISS_INDEX_DIR:-//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/faiss_index}"

# Create directories
SIGNAL_DIR="${OUTPUT_BASE_DIR}/${MODEL_ID}_v${VERSION}/signals"
WORKER_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_ID}_v${VERSION}/workers"
FINAL_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_ID}_v${VERSION}"
LOG_DIR="${PROJECT_ROOT}/logs_evidencerl"

mkdir -p "${SIGNAL_DIR}" "${WORKER_OUTPUT_DIR}" "${FINAL_OUTPUT_DIR}" "${LOG_DIR}"

# Calculate patients per worker
PATIENTS_PER_WORKER=$((TOTAL_PATIENTS / NUM_WORKERS))

echo "============================================================"
echo "EvidenceRL Generation Master"
echo "============================================================"
echo "MODEL_ID:        ${MODEL_ID}"
echo "MODEL_NAME:      ${MODEL_NAME}"
echo "VERSION:         ${VERSION}"
echo "TOTAL_PATIENTS:  ${TOTAL_PATIENTS}"
echo "NUM_WORKERS:     ${NUM_WORKERS}"
echo "PATIENTS/WORKER: ~${PATIENTS_PER_WORKER}"
echo "WORKER_RESOURCES: ${NUM_GPUS}x GPU, ${WORKER_CPUS} CPUs, ${WORKER_MEM}, ${WORKER_TIME}"
echo "OUTPUT_DIR:      ${FINAL_OUTPUT_DIR}"
echo "============================================================"

# Initialize conda for running merge script later
# Check common locations and PACE cluster-specific path
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
# Temporarily disable nounset for conda activation (some activation scripts have unbound vars)
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
    local end_idx=$(((worker_id + 1) * PATIENTS_PER_WORKER))

    # Handle last worker (may have extra patients)
    if [ $worker_id -eq $((NUM_WORKERS - 1)) ]; then
        end_idx=$TOTAL_PATIENTS
    fi

    # Build exclude node list
    local exclude_arg=""
    if [ ${#EXCLUDED_NODES[@]} -gt 0 ]; then
        exclude_arg="--exclude=$(IFS=,; echo "${EXCLUDED_NODES[*]}")"
    fi

    # Build resource override arguments based on configuration
    # Default worker sbatch has: 2x H200, 128GB RAM, 16 CPUs, 4h
    # These can be overridden via WORKER_* environment variables
    local resource_args="--gres=gpu:h200:${NUM_GPUS} --mem=${WORKER_MEM} --cpus-per-task=${WORKER_CPUS} --time=${WORKER_TIME}"

    echo "[master] Submitting worker ${worker_id} for patients [${start_idx}:${end_idx}]..."

    local job_id
    job_id=$(sbatch --parsable ${exclude_arg} ${resource_args} \
        --export=ALL,MODEL_ID="${MODEL_ID}",MODEL_NAME="${MODEL_NAME}",PATIENT_START="${start_idx}",PATIENT_END="${end_idx}",WORKER_ID="${worker_id}",VERSION="${VERSION}",OUTPUT_BASE_DIR="${WORKER_OUTPUT_DIR}",SIGNAL_DIR="${SIGNAL_DIR}",CONDA_ENV="${CONDA_ENV}",FAISS_INDEX_DIR="${FAISS_INDEX_DIR}",NUM_GPUS="${NUM_GPUS}" \
        "${WORKER_SCRIPT}")

    WORKER_JOB_IDS[$worker_id]=$job_id
    echo "[master] Worker ${worker_id} submitted with job ID: ${job_id}"
}

# Function to check if worker completed (via signal file or output file)
is_worker_complete() {
    local worker_id=$1
    local signal_file="${SIGNAL_DIR}/${MODEL_ID}_worker${worker_id}.done"
    local output_file="${WORKER_OUTPUT_DIR}/${MODEL_ID}_worker${worker_id}_v${VERSION}.json"

    # Check signal file first (fastest)
    if [ -f "$signal_file" ]; then
        return 0
    fi

    # Fallback: check if output file exists and is valid JSON
    if [ -f "$output_file" ]; then
        # Verify it's valid JSON with results
        if python3 -c "import json; d=json.load(open('${output_file}')); assert len(d.get('results',[])) > 0" 2>/dev/null; then
            # Create signal file for faster future checks
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
for ((i=0; i<NUM_WORKERS; i++)); do
    WORKER_RETRIES[$i]=0
    submit_worker $i
    sleep 1  # Small delay between submissions
done

echo ""
echo "[master] All workers submitted. Starting monitoring loop..."
echo "[master] Poll interval: ${POLL_INTERVAL} seconds"
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
                    # Job marked complete by Slurm - verify output exists
                    if is_worker_complete $i; then
                        ((++completed_count))
                    else
                        # Slurm says complete but no output - treat as failed
                        echo "[master] Warning: Worker ${i} marked COMPLETED but no valid output found"
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
                    # Job not found in sacct yet (just submitted), treat as pending
                    ((++pending_count))
                    ;;
                *)
                    # Unknown state, check again
                    ((++pending_count))
                    ;;
            esac
        fi
    done

    # Progress report
    echo "[master] Status: ${completed_count}/${NUM_WORKERS} complete, ${running_count} running, ${pending_count} pending, ${#failed_workers[@]} failed"

    # Handle failed workers
    for worker_id in "${failed_workers[@]}"; do
        retry_count=${WORKER_RETRIES[$worker_id]}
        if [ $retry_count -lt $MAX_RETRIES ]; then
            failed_job_id="${WORKER_JOB_IDS[$worker_id]}"
            failed_node=$(get_failed_node "$failed_job_id")

            echo "[master] Worker ${worker_id} failed (attempt $((retry_count + 1))/${MAX_RETRIES})"

            # Add failed node to exclusion list
            if [ -n "$failed_node" ] && [[ ! " ${EXCLUDED_NODES[*]} " =~ " ${failed_node} " ]]; then
                EXCLUDED_NODES+=("$failed_node")
                echo "[master] Adding node ${failed_node} to exclusion list"
            fi

            # Retry
            WORKER_RETRIES[$worker_id]=$((retry_count + 1))
            submit_worker $worker_id
        else
            echo "[master] Worker ${worker_id} exhausted all retries (${MAX_RETRIES})"
        fi
    done

    # Check if all done
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

FINAL_OUTPUT="${FINAL_OUTPUT_DIR}/${MODEL_ID}_${TOTAL_PATIENTS}-v${VERSION}.json"

python "${MERGE_SCRIPT}" \
    --input-dir "${WORKER_OUTPUT_DIR}" \
    --output "${FINAL_OUTPUT}" \
    --model-id "${MODEL_ID}" \
    --version "${VERSION}"

if [ -f "${FINAL_OUTPUT}" ]; then
    echo "[master] Merged results saved to: ${FINAL_OUTPUT}"

    # Count results
    result_count=$(python -c "import json; print(len(json.load(open('${FINAL_OUTPUT}'))['results']))")
    echo "[master] Total patients in merged output: ${result_count}"
else
    echo "[master] ERROR: Merge failed. Check individual worker outputs in: ${WORKER_OUTPUT_DIR}"
    exit 1
fi

echo ""
echo "============================================================"
echo "EvidenceRL Generation Complete"
echo "============================================================"
echo "Model:   ${MODEL_ID}"
echo "Output:  ${FINAL_OUTPUT}"
echo "Workers: ${OUTPUT_BASE_DIR}/${MODEL_ID}_v${VERSION}/workers/"
echo "============================================================"
