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
import sys # Added for sys.exit
from outetts.dac.interface import DacInterface

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')
logger = logging.getLogger(__name__)

# ===== Parameters to modify =====
TEXT = "مرحبا، أنا نموذج توليد صوت يمكنني نطق اللغة العربية بشكل طبيعي."  # Longer Arabic text
OUTPUT_FILE = "/content/direct_test_output.wav" # Ensure output to /content for Colab
MODEL_PATH = "OuteAI/Llama-OuteTTS-1.0-1B"  # Base model without LoRA
MAX_SEQ_LENGTH = 2048 # Max sequence length for the model
# ===============================

def main():
    # Initialize model with FastModel, exactly like the notebook
    logger.info(f"Loading base model: {MODEL_PATH}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Let Unsloth choose optimal dtype, was torch.float32
        load_in_4bit=False, # Base model inference, not 4-bit
    )
    logger.info(f"Model and tokenizer loaded. Model dtype: {model.dtype}")
    
    # Ensure model is on CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model moved to device: {device}")

    # Initialize DAC
    logger.info("Initializing DAC...")
    try:
        dac_interface = DacInterface(device=device)
        logger.info(f"DAC initialized on device: {dac_interface.device}, sample rate: {dac_interface.sr}")
    except Exception as e:
        logger.error(f"Failed to initialize DAC: {e}")
        sys.exit(1)

    # Prepare prompt (simplified, matching generate.py and notebook)
    text_to_generate = TEXT.strip()
    if not text_to_generate:
        logger.error("Input text is empty.")
        sys.exit(1)

    logger.info(f"Processing input text: '{text_to_generate}'")
    formatted_text = f"<|text_start|>{text_to_generate}<|text_end|>"
    prompt_text_part = "\n".join([
        "<|im_start|>",
        formatted_text,
        "<|audio_start|><|global_features_start|>",
    ])
    logger.info(f"Constructed prompt (first 150 chars): {prompt_text_part[:150]}...")
    logger.debug(f"Full prompt being tokenized:\n{prompt_text_part}")

    model_inputs = tokenizer([prompt_text_part], return_tensors="pt").to(device)
    input_ids_length = model_inputs.input_ids.shape[1]
    logger.info(f"Input prompt length in tokens: {input_ids_length}")

    # Generation parameters
    max_new_tokens = 2048 # As used in generate.py
    temperature = 0.4
    top_k = 40
    top_p = 0.9
    repetition_penalty = 1.1
    min_p = 0.05
    
    # Determine pad_token_id for generation
    gen_pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    logger.info(f"Using pad_token_id for generation: {gen_pad_token_id}")
    logger.info(f"Generating with max_new_tokens: {max_new_tokens}")
    logger.info(f"Generation params: temp={temperature}, top_k={top_k}, top_p={top_p}, rep_penalty={repetition_penalty}, min_p={min_p}")
    logger.info(f"Calling model.generate() with input_ids_length: {input_ids_length}, max_new_tokens: {max_new_tokens}")

    # Generate tokens
    logger.info("Generating token sequence...")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            max_new_tokens=max_new_tokens, 
            pad_token_id=gen_pad_token_id,
        )
    logger.info("Token sequence generated.")
    logger.info(f"Generated token IDs shape: {generated_ids.shape}")
    logger.info(f"Total generated sequence length (prompt + new tokens): {generated_ids.shape[1]}")
    logger.info(f"Generated token IDs (first 50): {generated_ids[0, :50].tolist()}")

    # Decode the *entire* output for token extraction
    decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    logger.info(f"Full decoded output length: {len(decoded_output)}")
    logger.info(f"Decoded output (first 300 chars): {decoded_output[:300]}")
    logger.info(f"Decoded output (last 300 chars): {decoded_output[-300:] if len(decoded_output) > 300 else decoded_output}")
    
    # Extract audio tokens
    c1_str_matches = re.findall(r"<\\|c1_(\\d*)\\|>", decoded_output)
    c2_str_matches = re.findall(r"<\\|c2_(\\d*)\\|>", decoded_output)

    logger.info(f"Raw c1 string matches (first 10): {c1_str_matches[:10]}")
    logger.info(f"Raw c2 string matches (first 10): {c2_str_matches[:10]}")

    try:
        c1 = [int(s) for s in c1_str_matches if s]  # Filter out empty strings
        c2 = [int(s) for s in c2_str_matches if s]  # Filter out empty strings
        logger.info(f"Found {len(c1)} valid c1 tokens and {len(c2)} valid c2 tokens")
    except ValueError as e:
        logger.error(f"Error converting token strings to integers: {e}")
        logger.error(f"Problematic c1_str_matches: {c1_str_matches}")
        logger.error(f"Problematic c2_str_matches: {c2_str_matches}")
        all_special_tokens = re.findall(r"<\\|[^|]+\\|>", decoded_output)
        logger.info(f"All special tokens found on ValueError: {list(set(all_special_tokens))}")
        sys.exit(1) # Critical error
    
    if not c1 or not c2:
        logger.error("No valid c1 or c2 tokens found after attempting to filter empty matches.")
        logger.info(f"Raw c1 matches: {c1_str_matches}")
        logger.info(f"Raw c2 matches: {c2_str_matches}")
        all_special_tokens = re.findall(r"<\\|[^|]+\\|>", decoded_output)
        logger.info(f"All special tokens found when c1/c2 are empty: {list(set(all_special_tokens))}")
        logger.warning("Problematic decoded output (first 500 chars):")
        logger.warning(decoded_output[:500])
        logger.warning("Problematic decoded output (last 500 chars):")
        logger.warning(decoded_output[-500:])
        sys.exit(1) # Critical error

    t = min(len(c1), len(c2))
    if t < 10: # Increased threshold slightly, was t == 0
        logger.error(f"Too few audio tokens found (c1: {len(c1)}, c2: {len(c2)}). Minimum common tokens: {t}. Cannot generate audio.")
        all_special_tokens = re.findall(r"<\\|[^|]+\\|>", decoded_output)
        logger.info(f"All special tokens found when t < 10: {list(set(all_special_tokens))}")
        sys.exit(1) # Critical error
    
    logger.info(f"Using {t} tokens for DAC decoding.")
    c1 = c1[:t]
    c2 = c2[:t]

    if len(c1) > 0:
        logger.info(f"First 5 c1 tokens: {c1[:5]}")
        logger.info(f"Last 5 c1 tokens: {c1[-5:]}")
    if len(c2) > 0:
        logger.info(f"First 5 c2 tokens: {c2[:5]}")
        logger.info(f"Last 5 c2 tokens: {c2[-5:]}")

    # Check DAC code range
    if any(code < 0 or code >= 1024 for code_list in [c1, c2] for code in code_list):
        logger.warning("Some DAC codes are out of the expected range [0, 1023].")
        # Log problematic codes for c1
        for i, code_val in enumerate(c1):
            if code_val < 0 or code_val >= 1024:
                logger.warning(f"c1[{i}] = {code_val} is out of range.")
        # Log problematic codes for c2
        for i, code_val in enumerate(c2):
            if code_val < 0 or code_val >= 1024:
                logger.warning(f"c2[{i}] = {code_val} is out of range.")

    output_codes = [c1, c2]
    
    try:
        with torch.no_grad():
            dac_input_tensor = torch.tensor([output_codes], dtype=torch.int64).to(dac_interface.device)
            logger.info(f"DAC input tensor shape: {dac_input_tensor.shape}, device: {dac_input_tensor.device}")
            audio_output = dac_interface.decode(dac_input_tensor)
            audio_output = audio_output.squeeze(0).cpu()
        logger.info(f"Audio generated by DAC, shape: {audio_output.shape}")
    except Exception as e:
        logger.error(f"Error during DAC decoding: {e}")
        logger.error(f"Problematic output_codes (first 5 of each): c1: {c1[:5]}, c2: {c2[:5]}")
        sys.exit(1)

    try:
        # Ensure output directory exists (especially for Colab /content)
        output_dir = os.path.dirname(OUTPUT_FILE)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
            
        sf.write(OUTPUT_FILE, audio_output, samplerate=dac_interface.sr)
        logger.info(f"Speech successfully generated and saved to {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Error saving audio file {OUTPUT_FILE}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 