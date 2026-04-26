# Source me once per shell:  source scripts/env.sh
#
# Routes the HuggingFace cache off AFS (which has flaky token-based read perms)
# and onto local scratch.

export HF_HOME=/scratch/azhang/.cache/huggingface
mkdir -p "$HF_HOME"
