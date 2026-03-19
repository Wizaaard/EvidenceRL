#!/usr/bin/env bash
#SBATCH -J erl_multisample_master
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH -o logs_multisample/multisample_master_%j.out
#SBATCH -e logs_multisample/multisample_master_%j.err

# ============================================================================
# EvidenceRL Multi-Sample Generation Master (vLLM)
# ============================================================================
# Orchestrates distributed multi-sample generation for per-model fdpo.
# Same architecture as baseline master but with --num-samples support.
# ============================================================================

set -euo pipefail

: "${MODEL_ID:?Need MODEL_ID}"
: "${MODEL_NAME:?Need MODEL_NAME}"
: "${VERSION:=1.0}"
: "${NUM_SAMPLES:=3}"

# Configuration
NUM_WORKERS="${NUM_WORKERS:-5}"
TOTAL_PATIENTS="${TOTAL_PATIENTS:-3700}"
PATIENT_OFFSET="${PATIENT_OFFSET:-0}"
MAX_RETRIES="${MAX_RETRIES:-3}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
MAX_TOKENS="${MAX_TOKENS:-2048}"

# Paths
PROJECT_ROOT="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/code"
OUTPUT_BASE_DIR="//storage/ice-shared/bmed-sp-wang/BenoitData/Restricted/evidenceRL/permodel_fdpo_samples/${MODEL_ID}"
WORKER_SCRIPT="${PROJECT_ROOT}/scripts/generation_multisample_vllm/multisample_generation_worker.sbatch"
MERGE_SCRIPT="${PROJECT_ROOT}/scripts/generation/merge_generation_results.py"

# Resources
NUM_GPUS="${NUM_GPUS:-2}"
WORKER_CPUS="${WORKER_CPUS:-16}"
WORKER_MEM="${WORKER_MEM:-128G}"
WORKER_TIME="${WORKER_TIME:-04:00:00}"

PATIENTS_PER_WORKER=$(( (TOTAL_PATIENTS + NUM_WORKERS - 1) / NUM_WORKERS ))

# Create directories
SIGNAL_DIR="${OUTPUT_BASE_DIR}/signals"
WORKER_OUTPUT_DIR="${OUTPUT_BASE_DIR}/workers"
mkdir -p "${OUTPUT_BASE_DIR}" "${SIGNAL_DIR}" "${WORKER_OUTPUT_DIR}" "${PROJECT_ROOT}/logs_multisample"

echo "============================================================"
echo "EvidenceRL Multi-Sample Generation Master (vLLM)"
echo "============================================================"
echo "MODE:            BASELINE (no-RAG) + ${NUM_SAMPLES} samples/patient"
echo "MODEL_ID:        ${MODEL_ID}"
echo "MODEL_NAME:      ${MODEL_NAME}"
echo "VERSION:         ${VERSION}"
echo "NUM_SAMPLES:     ${NUM_SAMPLES}"
echo "MAX_TOKENS:      ${MAX_TOKENS}"
echo "OUTPUT_DIR:      ${OUTPUT_BASE_DIR}"
echo "PATIENT_RANGE:   ${PATIENT_OFFSET} to $((PATIENT_OFFSET + TOTAL_PATIENTS)) (${TOTAL_PATIENTS} patients)"
echo "NUM_WORKERS:     ${NUM_WORKERS}"
echo "PATIENTS/WORKER: ~${PATIENTS_PER_WORKER}"
echo "EXTRACTOR:       ${EXTRACTOR_MODEL:-same as generator}"
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

CONDA_ENV="${CONDA_ENV:-ragcon}"
set +u
conda activate "${CONDA_ENV}" 2>/dev/null || true
set -u

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "${CONDA_PREFIX}" ]; then
    for conda_base in \
        "$HOME/scratch/conda_envs" \
        "$HOME/.conda/envs" \
        "$HOME/miniconda3/envs" \
        "$HOME/anaconda3/envs" \
        "/storage/ice1/8/6/$USER/conda_envs"; do
        if [ -d "${conda_base}/${CONDA_ENV}" ]; then
            export CONDA_PREFIX="${conda_base}/${CONDA_ENV}"
            break
        fi
    done
fi

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "${CONDA_PREFIX}" ]; then
    echo "[ERROR] Could not locate conda environment: ${CONDA_ENV}"
    exit 2
fi

echo "[multisample master] Using conda env: ${CONDA_ENV} at ${CONDA_PREFIX}"

if [ ! -x "${CONDA_PREFIX}/bin/python" ]; then
    echo "[ERROR] Python executable not found at: ${CONDA_PREFIX}/bin/python"
    exit 2
fi

# Track worker states
declare -A WORKER_JOB_IDS
declare -A WORKER_RETRIES
declare -a EXCLUDED_NODES=()

submit_worker() {
    local worker_id=$1
    local start_idx=$((PATIENT_OFFSET + worker_id * PATIENTS_PER_WORKER))
    local end_idx=$((PATIENT_OFFSET + (worker_id + 1) * PATIENTS_PER_WORKER))

    if [ $end_idx -gt $((PATIENT_OFFSET + TOTAL_PATIENTS)) ]; then
        end_idx=$((PATIENT_OFFSET + TOTAL_PATIENTS))
    fi

    if [ $start_idx -ge $((PATIENT_OFFSET + TOTAL_PATIENTS)) ]; then
        echo "[multisample master] Worker ${worker_id} has no patients, skipping"
        return 1
    fi

    local exclude_arg=""
    if [ ${#EXCLUDED_NODES[@]} -gt 0 ]; then
        exclude_arg="--exclude=$(IFS=,; echo "${EXCLUDED_NODES[*]}")"
    fi

    echo "[multisample master] Submitting worker ${worker_id} for patients [${start_idx}:${end_idx}]..."

    local job_id
    job_id=$(sbatch --parsable ${exclude_arg} \
        --gres=gpu:H100:${NUM_GPUS} \
        --cpus-per-task="${WORKER_CPUS}" \
        --mem="${WORKER_MEM}" \
        --time="${WORKER_TIME}" \
        --export=ALL,MODEL_ID="${MODEL_ID}",MODEL_NAME="${MODEL_NAME}",PATIENT_START="${start_idx}",PATIENT_END="${end_idx}",WORKER_ID="${worker_id}",VERSION="${VERSION}",WORKER_OUTPUT_DIR="${WORKER_OUTPUT_DIR}",SIGNAL_DIR="${SIGNAL_DIR}",CONDA_ENV="${CONDA_ENV}",NUM_GPUS="${NUM_GPUS}",EXTRACTOR_MODEL="${EXTRACTOR_MODEL:-}",NUM_SAMPLES="${NUM_SAMPLES}",MAX_TOKENS="${MAX_TOKENS}" \
        "${WORKER_SCRIPT}")

    WORKER_JOB_IDS[$worker_id]=$job_id
    echo "[multisample master] Worker ${worker_id} submitted: ${job_id}"
    return 0
}

is_worker_complete() {
    local worker_id=$1
    local signal_file="${SIGNAL_DIR}/${MODEL_ID}_multisample_worker${worker_id}.done"
    local json_output="${WORKER_OUTPUT_DIR}/${MODEL_ID}_multisample_worker${worker_id}_v${VERSION}.json"

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

get_job_state() {
    local job_id=$1
    sacct -j "$job_id" --format=State --parsable --noheader 2>/dev/null | head -1 | cut -d'|' -f1
}

get_failed_node() {
    local job_id=$1
    sacct -j "$job_id" --format=NodeList --parsable --noheader 2>/dev/null | head -1 | cut -d'|' -f1
}

# Submit all workers
echo ""
echo "[multisample master] Submitting ${NUM_WORKERS} worker jobs..."
ACTUAL_WORKERS=0
for ((i=0; i<NUM_WORKERS; i++)); do
    WORKER_RETRIES[$i]=0
    if submit_worker $i; then
        ((++ACTUAL_WORKERS))
    fi
    sleep 1
done

if [ $ACTUAL_WORKERS -eq 0 ]; then
    echo "[multisample master] ERROR: No workers submitted"
    exit 1
fi

echo ""
echo "[multisample master] ${ACTUAL_WORKERS} workers submitted. Monitoring..."
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
                *)
                    ((++pending_count))
                    ;;
            esac
        fi
    done

    echo "[multisample master] Status: ${completed_count}/${ACTUAL_WORKERS} complete, ${running_count} running, ${pending_count} pending, ${#failed_workers[@]} failed"

    # Handle failed workers
    for worker_id in "${failed_workers[@]}"; do
        retry_count=${WORKER_RETRIES[$worker_id]}
        if [ $retry_count -lt $MAX_RETRIES ]; then
            failed_job_id="${WORKER_JOB_IDS[$worker_id]}"
            failed_node=$(get_failed_node "$failed_job_id")

            echo "[multisample master] Worker ${worker_id} failed (attempt $((retry_count + 1))/${MAX_RETRIES})"

            if [ -n "$failed_node" ] && [[ ! " ${EXCLUDED_NODES[*]} " =~ " ${failed_node} " ]]; then
                EXCLUDED_NODES+=("$failed_node")
                echo "[multisample master] Excluding node: ${failed_node}"
            fi

            WORKER_RETRIES[$worker_id]=$((retry_count + 1))
            submit_worker $worker_id || true
        else
            echo "[multisample master] Worker ${worker_id} exhausted retries (${MAX_RETRIES})"
        fi
    done

    # All done?
    if [ $completed_count -eq $ACTUAL_WORKERS ]; then
        echo ""
        echo "[multisample master] All workers completed!"
        break
    fi

    # Check permanent failures
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
        echo "[multisample master] ${permanent_failures} workers permanently failed."
        echo "[multisample master] Proceeding with available results..."
        break
    fi

    sleep "${POLL_INTERVAL}"
done

# Merge results
echo ""
echo "[multisample master] Merging worker results..."

FINAL_OUTPUT="${OUTPUT_BASE_DIR}/${MODEL_ID}_multisample_${NUM_SAMPLES}x_${TOTAL_PATIENTS}patients_v${VERSION}.json"

${CONDA_PREFIX}/bin/python "${MERGE_SCRIPT}" \
    --input-dir "${WORKER_OUTPUT_DIR}" \
    --output "${FINAL_OUTPUT}" \
    --model-id "${MODEL_ID}" \
    --version "${VERSION}"

if [ -f "${FINAL_OUTPUT}" ]; then
    patient_count=$(${CONDA_PREFIX}/bin/python -c "import json; print(len(json.load(open('${FINAL_OUTPUT}'))['results']))")
    echo "[multisample master] Merged output: ${FINAL_OUTPUT}"
    echo "[multisample master] Total patients: ${patient_count}"
else
    echo "[multisample master] ERROR: Merge failed."
    exit 1
fi

echo ""
echo "============================================================"
echo "Multi-Sample Generation Complete"
echo "============================================================"
echo "Model:       ${MODEL_ID}"
echo "Samples:     ${NUM_SAMPLES} per patient"
echo "Output:      ${FINAL_OUTPUT}"
echo "Next step:   Score with NLI, then build fdpo pairs"
echo "============================================================"
