#!/usr/bin/env python3
"""
Direct test script for OuteTTS v1.0 model following the successful notebook pattern.
"""

from unsloth import FastModel
import torch
import re
import numpy as np
import logging
import soundfile as sf
import os
from outetts.dac.interface import DacInterface

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger(__name__)

# ===== Parameters to modify =====
TEXT = "مرحبا، أنا نموذج توليد صوت يمكنني نطق اللغة العربية بشكل طبيعي."  # Longer Arabic text
OUTPUT_FILE = "/content/direct_test_output.wav" # Ensure output to /content for Colab
MODEL_PATH = "OuteAI/Llama-OuteTTS-1.0-1B"  # Base model without LoRA
MAX_SEQ_LENGTH = 2048
# ===============================

def main():
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")

    # Initialize model with FastModel, exactly like the notebook
    logger.info(f"Loading model: {MODEL_PATH}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float32,  # Explicitly set to float32
        load_in_4bit=False,  # Don't use quantization for TTS
    )
    
    # Initialize DAC interface
    logger.info("Initializing DAC interface")
    dac_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"DAC interface device: {dac_device}")
    dac = DacInterface(device=dac_device)
    
    # Create input prompt
    formatted_text = f"<|text_start|>{TEXT}<|text_end|>"
    prompt = "\\n".join([
        "<|im_start|>",
        formatted_text,
        "<|audio_start|><|global_features_start|>",
    ])
    
    logger.info(f"Created prompt (first 100 chars): {prompt[:100]}...")
    
    # Tokenize input
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_length = model_inputs.input_ids.shape[1]
    logger.info(f"Input prompt length: {input_length} tokens")
    
    # Set fixed seed for reproducibility
    torch.manual_seed(3407)
    
    # Generate tokens
    logger.info("Generating token sequence...")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            temperature=0.4,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
            min_p=0.05,
            max_new_tokens=2048,  # Limit generation length
            # eos_token_id=tokenizer.eos_token_id, # Optional: Be explicit
            # pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id # Optional
        )
    logger.info("Token sequence generated.")
    
    # Extract and decode generated tokens
    generated_ids_trimmed = generated_ids[:, input_length:]
    logger.info(f"Generated IDs trimmed shape: {generated_ids_trimmed.shape}")
    logger.info(f"First 10 trimmed token IDs: {generated_ids_trimmed[0, :10].tolist()}")
    
    decoded_output = tokenizer.decode(generated_ids_trimmed[0], skip_special_tokens=False)
    
    # Debug generated output
    logger.info("---- Full Decoded Output (from generated_ids_trimmed) ----")
    # Limit printing if too long, but show a good chunk
    if len(decoded_output) > 2000:
        logger.info(decoded_output[:1000] + "\\n... (truncated) ...\\n" + decoded_output[-1000:])
    else:
        logger.info(decoded_output)
    logger.info("---- End of Full Decoded Output ----")
    
    # Extract audio tokens
    # Use \d* to capture zero or more digits, then filter empty strings before int conversion
    c1_str_matches = re.findall(r"<\\|c1_(\\d*)\\|>", decoded_output)
    c2_str_matches = re.findall(r"<\\|c2_(\\d*)\\|>", decoded_output)

    logger.info(f"Raw c1 string matches (first 10): {c1_str_matches[:10]}")
    logger.info(f"Raw c2 string matches (first 10): {c2_str_matches[:10]}")

    c1 = [int(s) for s in c1_str_matches if s]  # Filter out empty strings
    c2 = [int(s) for s in c2_str_matches if s]  # Filter out empty strings
    
    logger.info(f"Found {len(c1)} valid c1 tokens and {len(c2)} valid c2 tokens")
    
    if len(c1) > 0:
        logger.info(f"First 5 c1 tokens: {c1[:5]}")
        logger.info(f"Last 5 c1 tokens: {c1[-5:]}")
    if len(c2) > 0:
        logger.info(f"First 5 c2 tokens: {c2[:5]}")
        logger.info(f"Last 5 c2 tokens: {c2[-5:]}")

    if len(c1) < 10 or len(c2) < 10: # Increased threshold slightly
        logger.error(f"Too few audio tokens found (c1: {len(c1)}, c2: {len(c2)}). Cannot generate audio.")
        # Log all special tokens found if audio token extraction fails badly
        all_special_tokens = re.findall(r"<\\|[^|]+\\|>", decoded_output)
        logger.info(f"All special tokens found in decoded_output: {all_special_tokens[:50]}") # Log more
        return
    
    # Ensure equal length
    t = min(len(c1), len(c2))
    c1 = c1[:t]
    c2 = c2[:t]
    
    # Format for DAC
    audio_codes = [c1, c2]
    
    # Create tensor for DAC decoding
    codes_tensor = torch.tensor([audio_codes], dtype=torch.int64).to(dac.device)
    logger.info(f"Codes tensor shape: {codes_tensor.shape}")
    logger.info(f"Codes tensor (first 5 values per codebook): [[[c1_0..c1_4], [c2_0..c2_4]], ...]: {codes_tensor[:, :, :5].tolist()}")
    
    # Decode audio
    logger.info("Decoding audio...")
    try:
        audio_waveform = dac.decode(codes_tensor)
    except Exception as e:
        logger.error(f"Error during DAC decoding: {e}")
        logger.error(f"Codes tensor shape was: {codes_tensor.shape}, dtype: {codes_tensor.dtype}, device: {codes_tensor.device}")
        # Check DAC properties
        if hasattr(dac, 'model') and hasattr(dac.model, 'quantizer'):
            logger.error(f"DAC quantizer n_codebooks: {getattr(dac.model.quantizer, 'n_codebooks', 'N/A')}")
            logger.error(f"DAC quantizer codebook_size: {getattr(dac.model.quantizer, 'codebook_size', 'N/A')}")
        return

    if audio_waveform is None or audio_waveform.numel() == 0:
        logger.error("DAC decoding resulted in an empty tensor.")
        return
        
    logger.info(f"Audio waveform decoded. Shape: {audio_waveform.shape}, dtype: {audio_waveform.dtype}")

    # Convert to numpy and save
    audio_numpy = audio_waveform.squeeze().cpu().numpy()
    logger.info(f"Audio numpy array shape: {audio_numpy.shape}, dtype: {audio_numpy.dtype}")

    if audio_numpy.ndim == 0 or audio_numpy.size == 0:
        logger.error("Converted numpy audio is empty or scalar.")
        return

    # Ensure directory exists for output file (especially important in Colab)
    output_dir = os.path.dirname(os.path.abspath(OUTPUT_FILE))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
        
    # Save audio file
    logger.info(f"Saving audio to {OUTPUT_FILE} with sample rate 24000")
    sf.write(OUTPUT_FILE, audio_numpy, 24000) # OuteTTS standard sample rate is 24kHz
    logger.info("Done!")

if __name__ == "__main__":
    main() 