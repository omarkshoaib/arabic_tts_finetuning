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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Parameters to modify =====
TEXT = "مرحبا، أنا نموذج توليد صوت يمكنني نطق اللغة العربية بشكل طبيعي."  # Longer Arabic text
OUTPUT_FILE = "direct_test_output.wav"
MODEL_PATH = "OuteAI/Llama-OuteTTS-1.0-1B"  # Base model without LoRA
MAX_SEQ_LENGTH = 2048
# ===============================

def main():
    # Initialize model with FastModel, exactly like the notebook
    logger.info(f"Loading model: {MODEL_PATH}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Use default
        load_in_4bit=False,  # Don't use quantization for TTS
    )
    
    # Initialize DAC interface
    logger.info("Initializing DAC interface")
    dac = DacInterface(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Create input prompt
    formatted_text = f"<|text_start|>{TEXT}<|text_end|>"
    prompt = "\n".join([
        "<|im_start|>",
        formatted_text,
        "<|audio_start|><|global_features_start|>",
    ])
    
    logger.info(f"Created prompt: {prompt[:100]}...")
    
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
        )
    logger.info("Token sequence generated.")
    
    # Extract and decode generated tokens
    generated_ids_trimmed = generated_ids[:, input_length:]
    decoded_output = tokenizer.decode(generated_ids_trimmed[0], skip_special_tokens=False)
    
    # Debug generated output
    logger.info(f"Generated output (first 200 chars): {decoded_output[:200]}")
    
    # Extract audio tokens
    c1 = list(map(int, re.findall(r"<\|c1_(\d+)\|>", decoded_output)))
    c2 = list(map(int, re.findall(r"<\|c2_(\d+)\|>", decoded_output)))
    
    logger.info(f"Found {len(c1)} c1 tokens and {len(c2)} c2 tokens")
    
    if len(c1) == 0 or len(c2) == 0:
        logger.error("No audio tokens found!")
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
    
    # Decode audio
    logger.info("Decoding audio...")
    audio_waveform = dac.decode(codes_tensor)
    
    # Convert to numpy and save
    audio_numpy = audio_waveform.squeeze().cpu().numpy()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
    
    # Save audio file
    logger.info(f"Saving audio to {OUTPUT_FILE}")
    sf.write(OUTPUT_FILE, audio_numpy, 24000)
    logger.info("Done!")

if __name__ == "__main__":
    main() 