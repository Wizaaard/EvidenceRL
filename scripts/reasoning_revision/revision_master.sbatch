#!/usr/bin/env bash
#SBATCH -J erl_revision_master
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -o logs_evidencerl/revision_master_%j.out
#SBATCH -e logs_evidencerl/revision_master_%j.err

# ============================================================================
# EvidenceRL Reasoning Revision Master
# ============================================================================
# Orchestrates distributed reasoning revision by:
# 1. Reading generation and metrics files to determine patient count
# 2. Submitting worker jobs (each processes a subset of patients)
# 3. Monitoring worker completion via signal files
# 4. Retrying failed workers with node exclusion
# 5. Merging results when all workers complete
#
# Required environment variables:
#   GENERATION_JSON  - Path to generation JSON file
#   METRICS_JSON     - Path to metrics JSON file
#   OUTPUT_DIR       - Base output directory
#
# Optional environment variables:
#   MODEL_NAME       - Revision model path (default: medgemma-27b-it)
#   NUM_WORKERS      - Number of worker jobs (default: 10)
#   MAX_RETRIES      - Max retries per worker (default: 3)
#   POLL_INTERVAL    - Seconds between status checks (default: 60)
#   BATCH_SIZE       - Batch size for generation (default: 64)
#   GROUNDING_MIN    - Min grounding for ambiguous range (default: -0.25)
#   GROUNDING_MAX    - Max grounding for ambiguous range (default: 0.25)
#   SEED             - Random seed (default: 42)
# ============================================================================

set -euo pipefail

# Validate required environment variables
: "${GENERATION_JSON:?Need GENERATION_JSON}"
: "${METRICS_JSON:?Need METRICS_JSON}"
: "${OUTPUT_DIR:?Need OUTPUT_DIR}"

# Configuration
NUM_WORKERS="${NUM_WORKERS:-10}"
MAX_RETRIES="${MAX_RETRIES:-3}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"

# Paths
PROJECT_ROOT="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code"
MODEL_DIR="//storage/ice-shared/bmed-sp-wang/Models"
WORKER_SCRIPT="${PROJECT_ROOT}/scripts/reasoning_revision/revision_worker.sbatch"
MERGE_SCRIPT="${PROJECT_ROOT}/scripts/reasoning_revision/merge_revision_results.py"

# Model defaults
MODEL_NAME="${MODEL_NAME:-${MODEL_DIR}/medgemma-27b-it}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GROUNDING_MIN="${GROUNDING_MIN:--0.25}"
GROUNDING_MAX="${GROUNDING_MAX:-0.25}"
SEED="${SEED:-42}"

# Create directories
SIGNAL_DIR="${OUTPUT_DIR}/signals"
WORKER_OUTPUT_DIR="${OUTPUT_DIR}/workers"
LOG_DIR="${PROJECT_ROOT}/logs_evidencerl"

mkdir -p "${SIGNAL_DIR}" "${WORKER_OUTPUT_DIR}" "${LOG_DIR}"

# Get patient count from generation file
echo "[master] Reading generation file: ${GENERATION_JSON}"
TOTAL_PATIENTS=$(python3 -c "import json; print(len(json.load(open('${GENERATION_JSON}'))['results']))")
echo "[master] Found ${TOTAL_PATIENTS} patients in generation output"

# Calculate patients per worker
PATIENTS_PER_WORKER=$(( (TOTAL_PATIENTS + NUM_WORKERS - 1) / NUM_WORKERS ))

echo "============================================================"
echo "EvidenceRL Reasoning Revision Master"
echo "============================================================"
echo "GENERATION_JSON:  ${GENERATION_JSON}"
echo "METRICS_JSON:     ${METRICS_JSON}"
echo "OUTPUT_DIR:       ${OUTPUT_DIR}"
echo "MODEL_NAME:       ${MODEL_NAME}"
echo "TOTAL_PATIENTS:   ${TOTAL_PATIENTS}"
echo "NUM_WORKERS:      ${NUM_WORKERS}"
echo "PATIENTS/WORKER:  ~${PATIENTS_PER_WORKER}"
echo "BATCH_SIZE:       ${BATCH_SIZE}"
echo "GROUNDING_RANGE:  [${GROUNDING_MIN}, ${GROUNDING_MAX}]"
echo "SEED:             ${SEED}"
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
        --export=ALL,GENERATION_JSON="${GENERATION_JSON}",METRICS_JSON="${METRICS_JSON}",OUTPUT_DIR="${WORKER_OUTPUT_DIR}",MODEL_NAME="${MODEL_NAME}",PATIENT_START="${start_idx}",PATIENT_END="${end_idx}",WORKER_ID="${worker_id}",BATCH_SIZE="${BATCH_SIZE}",GROUNDING_MIN="${GROUNDING_MIN}",GROUNDING_MAX="${GROUNDING_MAX}",SEED="${SEED}",SIGNAL_DIR="${SIGNAL_DIR}",CONDA_ENV="${CONDA_ENV}" \
        "${WORKER_SCRIPT}")

    WORKER_JOB_IDS[$worker_id]=$job_id
    echo "[master] Worker ${worker_id} submitted with job ID: ${job_id}"
    return 0
}

# Function to check if worker completed
is_worker_complete() {
    local worker_id=$1
    local signal_file="${SIGNAL_DIR}/revision_worker${worker_id}.done"
    local gen_output="${WORKER_OUTPUT_DIR}/generation_worker${worker_id}.json"

    if [ -f "$signal_file" ]; then
        return 0
    fi

    if [ -f "$gen_output" ]; then
        if python3 -c "import json; d=json.load(open('${gen_output}')); assert 'results' in d" 2>/dev/null; then
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

FINAL_GEN_OUTPUT="${OUTPUT_DIR}/revised_generation.json"
FINAL_METRICS_OUTPUT="${OUTPUT_DIR}/revised_metrics.json"

python "${MERGE_SCRIPT}" \
    --input-dir "${WORKER_OUTPUT_DIR}" \
    --output-gen "${FINAL_GEN_OUTPUT}" \
    --output-metrics "${FINAL_METRICS_OUTPUT}" \
    --num-workers "${ACTUAL_WORKERS}"

if [ -f "${FINAL_GEN_OUTPUT}" ] && [ -f "${FINAL_METRICS_OUTPUT}" ]; then
    echo "[master] Merged generation saved to: ${FINAL_GEN_OUTPUT}"
    echo "[master] Merged metrics saved to: ${FINAL_METRICS_OUTPUT}"

    patient_count=$(python -c "import json; print(len(json.load(open('${FINAL_GEN_OUTPUT}'))['results']))")
    echo "[master] Total patients in merged output: ${patient_count}"
else
    echo "[master] ERROR: Merge failed. Check individual worker outputs in: ${WORKER_OUTPUT_DIR}"
    exit 1
fi

# Generate visualization plots
echo ""
echo "[master] Generating revision visualization plots..."

PLOT_SCRIPT="${PROJECT_ROOT}/scripts/reasoning_revision/plot_revision_results.py"
PLOTS_DIR="${OUTPUT_DIR}/plots"

if [ -f "${PLOT_SCRIPT}" ]; then
    python "${PLOT_SCRIPT}" \
        --metrics-json "${FINAL_METRICS_OUTPUT}" \
        --output-dir "${PLOTS_DIR}"

    if [ -d "${PLOTS_DIR}" ]; then
        echo "[master] Plots saved to: ${PLOTS_DIR}"
        echo "[master] Generated plots:"
        ls -la "${PLOTS_DIR}"/*.png 2>/dev/null || echo "[master] No PNG files found"
    else
        echo "[master] Warning: Plot generation may have failed. Check logs for details."
    fi
else
    echo "[master] Warning: Plot script not found at ${PLOT_SCRIPT}. Skipping visualization."
fi

echo ""
echo "============================================================"
echo "EvidenceRL Reasoning Revision Complete"
echo "============================================================"
echo "Output:  ${OUTPUT_DIR}"
echo "  - Generation: ${FINAL_GEN_OUTPUT}"
echo "  - Metrics:    ${FINAL_METRICS_OUTPUT}"
echo "  - Plots:      ${PLOTS_DIR}/"
echo "Workers: ${WORKER_OUTPUT_DIR}/"
echo "============================================================"
