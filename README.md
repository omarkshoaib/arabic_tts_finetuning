# Arabic Sales TTS - OuteTTS Finetuning

This project fine-tunes the OuteTTS v1.0 model specifically for Arabic sales speech synthesis using Unsloth optimization.

## Project Overview

OuteTTS is a text-to-speech model that can generate high-quality natural speech. This project adapts OuteTTS v1.0 specifically for Arabic sales conversations by fine-tuning on domain-specific data.

Key features:
- Fine-tuning OuteTTS v1.0 (1B parameter model) on Arabic data
- Using Unsloth optimization for efficient training with lower VRAM requirements
- Processing audio with Whisper for proper word-level alignment 
- Supporting speaker identification for multi-speaker datasets

## Directory Structure

```
arabic_tts_finetuning/
├── configs/             # Configuration files
├── data/                # Data utilities and preprocessing
├── models/              # Model training and inference
├── src/                 # Core source code
│   ├── preprocessing/   # Audio and text preprocessing utilities
│   ├── training/        # Training scripts
│   └── inference/       # Inference scripts
└── requirements.txt     # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/arabic-tts-finetuning.git
cd arabic-tts-finetuning
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have the following packages installed:
- torch
- unsloth
- transformers
- datasets
- whisper
- dac
- outetts
- polars

## Usage

### Data Preparation

1. Place your audio data in a structure with:
   - WAV files in a `wavs` directory
   - A metadata CSV file with columns: `audio_file|text|speaker_name`

2. Run data preprocessing:
```bash
python src/preprocessing/process_dataset.py --data_dir /path/to/data --output_dir /path/to/output
```

### Training

```bash
python src/training/train.py --config configs/train_config.yaml
```

### Inference

```bash
python src/inference/generate.py --model_path /path/to/finetuned/model --text "يوجد شقة للبيع في القاهرة الجديدة بمساحة 120 متر مربع."
```

## Model Architecture

This project uses OuteTTS v1.0 (1B parameters) with the following enhancements:
- Whisper for audio transcription and word alignment
- DAC (Diffusion Audio Codec) instead of WavTokenizer used in older versions
- Interface V3 for modern voice generation
- Unsloth optimization for efficient training

## License

This project is licensed under the Apache 2.0 License. 