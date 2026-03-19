#!/bin/bash
# EvidenceRL Environment Setup Script
#
# Usage:
#   ./setup_env.sh              # Create environment with default name 'evidencerl'
#   ./setup_env.sh myenv        # Create environment with custom name
#   ./setup_env.sh --pip-only   # Use pip instead of conda
#
# Requirements:
#   - conda (Anaconda or Miniconda)
#   - CUDA 12.x (for GPU support)

set -e

ENV_NAME="${1:-evidencerl}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  EvidenceRL Environment Setup${NC}"
echo -e "${GREEN}============================================${NC}"

# Check if --pip-only flag is passed
if [ "$1" == "--pip-only" ]; then
    echo -e "${YELLOW}Using pip-only installation...${NC}"

    # Create virtual environment
    echo -e "\n${GREEN}Creating virtual environment...${NC}"
    python -m venv .venv
    source .venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install requirements
    echo -e "\n${GREEN}Installing requirements...${NC}"
    pip install -r "${SCRIPT_DIR}/requirements.txt"

    echo -e "\n${GREEN}============================================${NC}"
    echo -e "${GREEN}  Setup Complete!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo -e "\nActivate with: ${YELLOW}source .venv/bin/activate${NC}"
    exit 0
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install Anaconda or Miniconda.${NC}"
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check CUDA availability
echo -e "\n${GREEN}Checking CUDA...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo -e "${GREEN}CUDA detected!${NC}"
else
    echo -e "${YELLOW}Warning: nvidia-smi not found. GPU support may not work.${NC}"
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment '${ENV_NAME}' already exists.${NC}"
    read -p "Do you want to remove and recreate it? (y/N): " confirm
    if [ "$confirm" == "y" ] || [ "$confirm" == "Y" ]; then
        echo -e "${YELLOW}Removing existing environment...${NC}"
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "Exiting. Use a different environment name or activate the existing one."
        exit 0
    fi
fi

# Create conda environment
echo -e "\n${GREEN}Creating conda environment '${ENV_NAME}'...${NC}"
echo "This may take 10-20 minutes depending on your internet connection."

conda env create -f "${SCRIPT_DIR}/environment.yml" -n "${ENV_NAME}"

# Activate and verify
echo -e "\n${GREEN}Verifying installation...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Verify key packages
echo -e "\nVerifying packages:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
python -c "import sentence_transformers; print(f'  Sentence-Transformers: {sentence_transformers.__version__}')"
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')"

# Check FAISS
python -c "import faiss; print(f'  FAISS: OK (GPU: {faiss.get_num_gpus() > 0})')" 2>/dev/null || echo "  FAISS: Not installed or failed"

# Add src to PYTHONPATH hint
echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "\nActivate the environment:"
echo -e "  ${YELLOW}conda activate ${ENV_NAME}${NC}"
echo -e "\nTo add the source to your PYTHONPATH:"
echo -e "  ${YELLOW}export PYTHONPATH=${SCRIPT_DIR}/src:\$PYTHONPATH${NC}"
echo -e "\nOr add this to your ~/.bashrc for permanent access."
echo -e "\nTo login to HuggingFace (required for gated models):"
echo -e "  ${YELLOW}huggingface-cli login${NC}"
