#!/bin/bash
#------------------------------------------------------------
# TTU HPCC — RedRaider Matador GPU partition
# Runs ONLY llama3.3:70b generation (all 817 questions).
# Evaluation and merging are done separately in Google Colab.
#
# Submit with:  sbatch hpcc_job.sh
# Monitor with: squeue -u $USER
# Output log:   slurm-<jobid>.out
#------------------------------------------------------------

#SBATCH --job-name=hallucination-70b
#SBATCH --partition=matador
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emabuyak@ttu.edu

echo "======================================================"
echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "======================================================"

# ── Paths ──────────────────────────────────────────────────
PROJECT_DIR="$HOME/llm-hallucination-phoenix-main"
VENV_DIR="$PROJECT_DIR/.venv"
OLLAMA_BIN="$HOME/.ollama/bin/ollama"
OLLAMA_MODELS_DIR="$HOME/.ollama/models"
OLLAMA_HOST="127.0.0.1"
OLLAMA_PORT="11434"
OLLAMA_BASE_URL="http://${OLLAMA_HOST}:${OLLAMA_PORT}"
MODEL="llama3.3:70b"

# ── Activate virtual environment ───────────────────────────
echo "[1/4] Activating Python virtual environment..."
source "$VENV_DIR/bin/activate"
python --version

# ── Start Ollama server ────────────────────────────────────
echo "[2/4] Starting Ollama server..."
export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"
export OLLAMA_HOST="${OLLAMA_HOST}:${OLLAMA_PORT}"
export OLLAMA_NUM_PARALLEL=4
export CUDA_VISIBLE_DEVICES=0,1

"$OLLAMA_BIN" serve &
OLLAMA_PID=$!
echo "  Ollama PID: $OLLAMA_PID"

# Wait for Ollama to be ready
echo "  Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
    if curl -s "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; then
        echo "  Ollama is ready."
        break
    fi
    sleep 2
done

# ── Pull model ─────────────────────────────────────────────
echo "[3/4] Pulling $MODEL (skipped if already cached)..."
"$OLLAMA_BIN" pull "$MODEL"
echo "  Model ready."
"$OLLAMA_BIN" list

# ── Warm up model before generation ───────────────────────
echo "  Warming up $MODEL — waiting for first response..."
WARMUP_OK=0
for i in $(seq 1 40); do
    RESPONSE=$(curl -s --max-time 120 "${OLLAMA_BASE_URL}/api/chat" \
        -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one word: ready\"}],\"stream\":false,\"options\":{\"num_predict\":5}}" \
        2>/dev/null)
    if echo "$RESPONSE" | grep -q "content"; then
        echo "  Model warmed up after $((i * 15))s."
        WARMUP_OK=1
        break
    fi
    echo "  Not ready yet (attempt $i/40)... retrying in 15s"
    sleep 15
done

if [ "$WARMUP_OK" -eq 0 ]; then
    echo "ERROR: $MODEL did not respond within 10 minutes. Aborting."
    kill "$OLLAMA_PID" 2>/dev/null
    exit 1
fi

# ── Run generation ─────────────────────────────────────────
echo "[4/4] Running generation for $MODEL (all 817 questions)..."
echo "  Checkpoint/resume active — safe to resubmit if job times out."
echo "  Started: $(date)"
cd "$PROJECT_DIR"
python src/run_experiment.py
echo "  Generation done: $(date)"

# ── Cleanup ────────────────────────────────────────────────
echo "Stopping Ollama server (PID $OLLAMA_PID)..."
kill "$OLLAMA_PID" 2>/dev/null

echo "======================================================"
echo "Job finished : $(date)"
echo "Output file  : $PROJECT_DIR/data/experiment_results.csv"
ROWS=$(python -c "import pandas as pd; df=pd.read_csv('$PROJECT_DIR/data/experiment_results.csv'); print(len(df))" 2>/dev/null || echo "unknown")
echo "Rows written : $ROWS"
echo "======================================================"
echo "Next step: download experiment_results.csv and merge"
echo "with existing 3-model data in Google Colab."
echo "======================================================"
