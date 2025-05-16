#!/bin/bash
# Run script for Arabic TTS fine-tuning project

set -e  # Exit on error

# cd to the script's own directory to make relative paths work reliably
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

# --- Logging ---
log_message() {
  local type="$1"
  local message="$2"
  echo "$(date '+%Y-%m-%d %H:%M:%S') | $type | $message"
}
# --- End Logging ---

# Default values
MODE="all"  # all, preprocess, train, or inference
DATA_DIR="/content/drive/MyDrive/Address/data_ottus" # Default for Colab, can be overridden
PROCESSED_DIR="./data/processed"
MODELS_DIR="./models/arabic_sales_tts"
OUTPUT_DIR="./outputs"
TRAIN_CONFIG="./configs/train_config.yaml"
INFERENCE_CONFIG="./configs/inference_config.yaml"
TEXT="" # For inference
OUTPUT_FILE="" # For inference

# Variables for preprocessing, with defaults matching process_dataset.py
METADATA_FILE_NAME="metadata_train.csv" # Default original metadata file
PROCESSED_DIR_NAME="processed"
MODEL_TOKENIZER_PATH="OuteAI/Llama-OuteTTS-1.0-1B"
WHISPER_MODEL="medium"
BATCH_SIZE="32" # Default batch_size for preprocessing

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --processed-dir)
      PROCESSED_DIR="$2"
      shift 2
      ;;
    --models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --train-config)
      TRAIN_CONFIG="$2"
      shift 2
      ;;
    --inference-config)
      INFERENCE_CONFIG="$2"
      shift 2
      ;;
    --text)
      TEXT="$2"
      shift 2
      ;;
    --output-file)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --metadata-file)
      METADATA_FILE_NAME="$2"
      shift 2
      ;;
    --model-tokenizer) # Corresponds to MODEL_TOKENIZER_PATH for the script
      MODEL_TOKENIZER_PATH="$2"
      shift 2
      ;;
    --whisper-model)
      WHISPER_MODEL="$2"
      shift 2
      ;;
    --batch-size) # For preprocessing batch_size
      BATCH_SIZE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --mode MODE            Run mode: all, preprocess, train, or inference (default: all)"
      echo "  --data-dir DIR         Path to raw data directory (default: ./data_ottus-20250505T133830Z-001/data_ottus)"
      echo "  --processed-dir DIR    Path to store processed data (default: ./data/processed)"
      echo "  --models-dir DIR       Path to store/load models (default: ./models/arabic_sales_tts)"
      echo "  --output-dir DIR       Path to store outputs (default: ./outputs)"
      echo "  --train-config FILE    Path to training config file (default: ./configs/train_config.yaml)"
      echo "  --inference-config FILE Path to inference config file (default: ./configs/inference_config.yaml)"
      echo "  --text TEXT            Text for inference (only for inference mode)"
      echo "  --output-file FILE     Output audio file path (only for inference mode)"
      echo "  --metadata-file FILE   Metadata CSV file name (default: metadata_train.csv, for preprocessing)"
      echo "  --model-tokenizer PATH Model tokenizer path for preprocessing (default: OuteAI/Llama-OuteTTS-1.0-1B)"
      echo "  --whisper-model MODEL  Whisper model name for preprocessing (default: medium)"
      echo "  --batch-size NUM       Batch size for preprocessing (default: 32)"
      echo "  --help                 Show this help message and exit"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Function to create necessary directories
create_dirs() {
  log_message "INFO" "Ensuring base directories exist..."

  # Validate and create PROCESSED_DIR
  if [ -z "$PROCESSED_DIR" ]; then
    log_message "ERROR" "PROCESSED_DIR is not set. Cannot create directory."
    exit 1
  fi
  PROCESSED_DIR_BASE=$(dirname "$PROCESSED_DIR")
  if [ -n "$PROCESSED_DIR_BASE" ] && [ "$PROCESSED_DIR_BASE" != "." ] && [ "$PROCESSED_DIR_BASE" != "/" ]; then
    mkdir -p "$PROCESSED_DIR_BASE"
  fi
  mkdir -p "$PROCESSED_DIR"

  # Validate and create MODELS_DIR
  if [ -z "$MODELS_DIR" ]; then
    log_message "ERROR" "MODELS_DIR is not set. Cannot create directory."
    exit 1
  fi
  MODELS_DIR_BASE=$(dirname "$MODELS_DIR")
  if [ -n "$MODELS_DIR_BASE" ] && [ "$MODELS_DIR_BASE" != "." ] && [ "$MODELS_DIR_BASE" != "/" ]; then
    mkdir -p "$MODELS_DIR_BASE"
  fi
  mkdir -p "$MODELS_DIR"

  # Validate and create OUTPUT_DIR
  if [ -z "$OUTPUT_DIR" ]; then
    log_message "ERROR" "OUTPUT_DIR is not set. Cannot create directory."
    exit 1
  fi
  mkdir -p "$OUTPUT_DIR"

  log_message "INFO" "Base directories ensured."
}

# Function for preprocessing
run_preprocess() {
  echo "===== PREPROCESSING DATA ====="
  # DATA_DIR is set from run.sh CLI arguments or default.
  # PROCESSED_DIR is globally set to ./data/processed
  # METADATA_FILE_NAME is set (e.g., metadata_train_cleaned.csv)
  # MODEL_TOKENIZER_PATH, WHISPER_MODEL, BATCH_SIZE are set.

  echo "Raw data base dir: ${DATA_DIR}"
  echo "Metadata file name to be found in raw data dir: ${METADATA_FILE_NAME}"
  echo "Processed data output dir: ${PROCESSED_DIR}"
  
  log_message "INFO" "Starting preprocessing..."
  
  # Construct full path to the metadata file
  METADATA_FULL_PATH="${DATA_DIR}/${METADATA_FILE_NAME}"

  python3 "${SCRIPT_DIR}/src/preprocessing/process_dataset.py" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${PROCESSED_DIR}" \
    --metadata_file "${METADATA_FULL_PATH}" \
    --model_tokenizer "${MODEL_TOKENIZER_PATH}" \
    --whisper_model "${WHISPER_MODEL}" \
    --batch_size "${BATCH_SIZE}"
  log_message "INFO" "Preprocessing finished."
}

# Function for training
run_train() {
  echo "===== TRAINING MODEL ====="
  echo "Processed data dir: $PROCESSED_DIR"
  echo "Models dir: $MODELS_DIR"
  echo "Config file: $TRAIN_CONFIG"
  
  # Update config file with correct paths
  sed -i "s|data_dir:.*|data_dir: \"$PROCESSED_DIR\"|g" "$TRAIN_CONFIG"
  sed -i "s|output_dir:.*|output_dir: \"$MODELS_DIR\"|g" "$TRAIN_CONFIG"
  
  python3 src/training/train.py \
    --config "$TRAIN_CONFIG"
  
  echo "Training complete!"
}

# Function for inference
run_inference() {
  echo "===== GENERATING SPEECH ====="
  echo "Models dir: $MODELS_DIR"
  echo "Output dir: $OUTPUT_DIR"
  echo "Config file: $INFERENCE_CONFIG"
  
  # Update config file with correct paths
  sed -i "s|model_path:.*|model_path: \"$MODELS_DIR\"|g" "$INFERENCE_CONFIG"
  
  # Handle optional parameters
  EXTRA_ARGS=""
  if [ -n "$TEXT" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --text \"$TEXT\""
  fi
  
  if [ -n "$OUTPUT_FILE" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --output_file \"$OUTPUT_FILE\""
  else
    # Use default output file
    OUTPUT_FILE="$OUTPUT_DIR/sample_output.wav"
    mkdir -p "$OUTPUT_DIR"
  fi
  
  python3 src/inference/generate.py \
    --config "$INFERENCE_CONFIG" \
    $EXTRA_ARGS
  
  echo "Inference complete! Output saved to: $OUTPUT_FILE"
}

# Call function to create directories first
create_dirs

# Execute based on mode
case $MODE in
  "all")
    run_preprocess
    run_train
    run_inference
    ;;
  "preprocess")
    run_preprocess
    ;;
  "train")
    run_train
    ;;
  "inference")
    run_inference
    ;;
  *)
    echo "Invalid mode: $MODE"
    echo "Valid modes: all, preprocess, train, inference"
    exit 1
    ;;
esac

echo "Done!" 