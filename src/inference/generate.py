#!/usr/bin/env python3
"""
Inference script for generating speech using the fine-tuned OuteTTS model (v1.0 / v3 interface).
"""

# Attempt to mitigate protobuf import errors by importing tensorflow first
# import tensorflow # Commenting out to test if it resolves segmentation fault

# Unsloth should be imported early
from unsloth import FastModel # Using Unsloth for consistency if fine-tuned with it

import os
import argparse
import torch
import logging
import yaml
import soundfile as sf
import numpy as np
from typing import Optional
from transformers import AutoTokenizer
from peft import PeftModel
from outetts.dac.interface import DacInterface
# from outetts.models.llama_tts import LlamaTTS # Old v0.3 import
from outetts.version.v3.prompt_processor import PromptProcessor # v3 import
from outetts.models.config import ModelConfig # For dummy config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OuteTTSGeneratorV3:
    def __init__(
        self,
        model_path: str, # Path to LoRA adapter if use_lora is True, else full model path
        use_lora: bool = True,
        base_model_name: str = "OuteAI/Llama-OuteTTS-1.0-1B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dac_ckpt: Optional[str] = None, # Usually not needed if OuteTTS handles it
        max_seq_length: int = 2048, # Should match training
    ):
        """
        Initialize the OuteTTS v3 generator.
        """
        self.device = device
        self.max_seq_length = max_seq_length
        self.use_lora = use_lora
        self.model_path = model_path
        self.base_model_name = base_model_name

        logger.info(f"Initializing OuteTTSGeneratorV3. Device: {self.device}, LoRA: {self.use_lora}")
        logger.info(f"Model path: {self.model_path}, Base model: {self.base_model_name}")

        if self.use_lora:
            logger.info(f"Loading base model '{self.base_model_name}' with LoRA adapter from '{self.model_path}'")
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=self.base_model_name,
                max_seq_length=self.max_seq_length,
                dtype=None, # Let Unsloth choose optimal dtype for base
                load_in_4bit=False, # Base model is not 4-bit when using LoRA
            )
            logger.info(f"Base model loaded with dtype: {self.model.dtype}")
            self.model = FastModel.get_peft_model(
                self.model,
                self.model_path,
                # r = 16, # default
                # lora_alpha = 16, # default
            )
            logger.info("LoRA adapter loaded and merged.")
        else:
            logger.info(f"Loading full model from '{self.model_path}' (no LoRA)")
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=self.model_path, # This is the base model itself when not using LoRA
                max_seq_length=self.max_seq_length,
                dtype=None, # Let Unsloth choose optimal dtype
                load_in_4bit=False, # Assuming base model inference is not 4-bit
            )
            logger.info(f"Full model loaded with dtype: {self.model.dtype}")

        self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")
        
        # Initialize DAC
        logger.info("Initializing DAC interface...")
        # The DacInterface expects device and optionally model_path (dac_ckpt),
        # not a config object.
        self.dac = DacInterface(device=self.device, model_path=dac_ckpt)
        
        try:
            self.dac_n_codebooks = self.dac.model.quantizer.n_codebooks
            self.dac_codebook_bins = self.dac.model.quantizer.codebook_size
            logger.info(f"DAC model initialized with n_codebooks: {self.dac_n_codebooks} and codebook_size (bins per quantizer): {self.dac_codebook_bins}")
        except AttributeError as e:
            logger.error(f"Failed to retrieve n_codebooks or codebook_size from DAC model's quantizer: {e}")
            logger.error("self.dac.model.quantizer or its attributes .n_codebooks/.codebook_size might be missing.")
            logger.error("This is critical for DAC token processing. Please check DAC model compatibility and initialization.")
            # Attempt to inspect the quantizer object if it exists
            if hasattr(self.dac, 'model') and hasattr(self.dac.model, 'quantizer'):
                logger.error(f"DAC quantizer object: {self.dac.model.quantizer}")
                logger.error(f"DAC quantizer attributes: {dir(self.dac.model.quantizer)}")
            else:
                logger.error("self.dac.model or self.dac.model.quantizer is not available for inspection.")
            raise ValueError("Critical DAC model attributes (n_codebooks, codebook_size) could not be determined.") from e

        # Initialize v3 PromptProcessor
        logger.info("Initializing v3 PromptProcessor...")
        self.prompt_processor = PromptProcessor(base_model_name) # Needs tokenizer path
        
    def generate_speech(
        self,
        text: str,
        output_file: str,
        speaker_name: str = "default_speaker", # Default speaker tag (not used in OuteTTS v1.0)
        lang: str = "ar", # Explicitly using "lang" as per v3 (not used in OuteTTS v1.0)
        top_p: float = 0.9,
        top_k: int = 40, 
        temperature: float = 0.4, 
        repetition_penalty: float = 1.1, 
        min_p: float = 0.05, 
        max_new_tokens: int = 2048, 
        sample_rate: int = 24000,
    ):
        """
        Generate speech from text using OuteTTS v1.0 model.
        Prompt structure and generation parameters are based on Oute_TTS_(1B).ipynb.
        """
        # Clean and normalize input text
        text = text.strip()
        if not text:
            logger.error("Input text is empty after cleaning")
            return False
            
        logger.info(f"Processing input text: '{text}'")
        
        # Build prompt based on Oute_TTS_(1B).ipynb
        # The notebook prompt ends after <|global_features_start|>
        # The model is expected to generate the global feature tokens and the rest.
        formatted_text = f"<|text_start|>{text}<|text_end|>"
        prompt_text_part = "\n".join([
            "<|im_start|>",
            formatted_text,
            "<|audio_start|><|global_features_start|>",
            # Removed: global_feature_tokens, "<|global_features_end|>", "<|word_start|>"
        ])
        
        logger.info(f"Constructed prompt (first 150 chars): {prompt_text_part[:150]}...")
        logger.debug(f"Full prompt being tokenized:\n{prompt_text_part}")
        
        inputs = self.tokenizer(prompt_text_part, return_tensors="pt").to(self.device)
        input_ids_length = inputs.input_ids.shape[1]
        logger.info(f"Input prompt length in tokens: {input_ids_length}")

        # Determine pad_token_id for generation
        # If tokenizer.pad_token_id is None, use tokenizer.eos_token_id
        # This is a common practice when pad_token_id is not explicitly set.
        gen_pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        logger.info(f"Using pad_token_id for generation: {gen_pad_token_id}")

        # Generation parameters based on the notebook
        logger.info(f"Generating with max_new_tokens: {max_new_tokens}")
        logger.info(f"Generation params: temp={temperature}, top_k={top_k}, top_p={top_p}, rep_penalty={repetition_penalty}, min_p={min_p}")
        logger.info(f"Calling model.generate() with input_ids_length: {input_ids_length}, max_new_tokens: {max_new_tokens}")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                max_new_tokens=max_new_tokens, # Max new tokens for the model to generate
                # eos_token_id=eos_id, # Removed: Notebook does not use explicit EOS for generation
                pad_token_id=gen_pad_token_id, # Use determined pad_token_id
                # bos_token_id=self.tokenizer.bos_token_id, # Usually not needed if prompt has BOS
                # no_repeat_ngram_size=0, # Default, can be tuned
                # num_beams=1, # Default (greedy/sampling)
            )
        
        logger.info(f"Generated token IDs shape: {generated_ids.shape}")
        logger.info(f"Total generated sequence length (prompt + new tokens): {generated_ids.shape[1]}")
        logger.info(f"Generated token IDs (first 50): {generated_ids[0, :50].tolist()}")

        # Decode the *entire* output (including prompt) as done in the notebook for token extraction
        decoded_output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        
        logger.info(f"Full decoded output length: {len(decoded_output)}")
        logger.info(f"Decoded output (first 300 chars): {decoded_output[:300]}")
        logger.info(f"Decoded output (last 300 chars): {decoded_output[-300:] if len(decoded_output) > 300 else decoded_output}")
        
        # OuteTTS v1.0 uses c1_XXX and c2_XXX format
        import re
        # Regex to find <|c1_DIGITS|> and <|c2_DIGITS|>
        # Use \d* to capture zero or more digits, then filter empty strings before int conversion
        c1_str_matches = re.findall(r"<|c1_(\d*)\||>", decoded_output)
        c2_str_matches = re.findall(r"<|c2_(\d*)\||>", decoded_output)

        logger.info(f"Raw c1 string matches (first 10): {c1_str_matches[:10]}")
        logger.info(f"Raw c2 string matches (first 10): {c2_str_matches[:10]}")

        try:
            c1 = [int(s) for s in c1_str_matches if s]  # Filter out empty strings and convert
            c2 = [int(s) for s in c2_str_matches if s]  # Filter out empty strings and convert
            logger.info(f"Found {len(c1)} valid c1 tokens and {len(c2)} valid c2 tokens")
        except ValueError as e:
            logger.error(f"Error converting token strings to integers: {e}")
            logger.error(f"Problematic c1_str_matches: {c1_str_matches}")
            logger.error(f"Problematic c2_str_matches: {c2_str_matches}")
            # Log all special tokens found if there's a ValueError, as it might indicate unexpected token format
            all_special_tokens = re.findall(r"<|[^|]+|>", decoded_output)
            logger.info(f"All special tokens found on error: {all_special_tokens}")
            return False
        
        if not c1 or not c2:
            logger.error("No valid c1 or c2 tokens found after attempting to filter empty matches.")
            logger.info(f"Raw c1 matches: {c1_str_matches}")
            logger.info(f"Raw c2 matches: {c2_str_matches}")
            all_special_tokens = re.findall(r"<|[^|]+|>", decoded_output)
            logger.info(f"All special tokens found when c1/c2 are empty: {list(set(all_special_tokens))}") # Using set to see unique tokens
            # Log a larger portion of the decoded output if token extraction fails
            logger.warning("Potentially problematic decoded output (first 500 chars):")
            logger.warning(decoded_output[:500])
            logger.warning("Potentially problematic decoded output (last 500 chars):")
            logger.warning(decoded_output[-500:])
            return False

        t = min(len(c1), len(c2))
        if t == 0:
            logger.error("No common tokens between c1 and c2. Cannot generate audio.")
            all_special_tokens = re.findall(r"<|[^|]+|>", decoded_output) # Log special tokens
            logger.info(f"All special tokens found when t=0: {list(set(all_special_tokens))}")
            return False
        
        logger.info(f"Using {t} tokens for DAC decoding.")
        c1 = c1[:t]
        c2 = c2[:t]
        
        if len(c1) > 0:
            logger.info(f"First 5 c1 tokens: {c1[:5]}")
            logger.info(f"Last 5 c1 tokens: {c1[-5:]}")
        if len(c2) > 0:
            logger.info(f"First 5 c2 tokens: {c2[:5]}")
            logger.info(f"Last 5 c2 tokens: {c2[-5:]}")

        # Check if DAC codes are within expected range (0-1023 for DAC)
        if any(code < 0 or code >= 1024 for code_list in [c1, c2] for code in code_list):
            logger.warning("Some DAC codes are out of the expected range [0, 1023]. This might indicate an issue.")
            # Log problematic codes
            for i, code in enumerate(c1):
                if code < 0 or code >= 1024:
                    logger.warning(f"c1[{i}] = {code} is out of range.")
            for i, code in enumerate(c2):
                if code < 0 or code >= 1024:
                    logger.warning(f"c2[{i}] = {code} is out of range.")
        
        output_codes = [c1, c2]
        
        try:
            with torch.no_grad():
                # Ensure codes are on the correct device for DAC model
                dac_input_tensor = torch.tensor([output_codes], dtype=torch.int64).to(self.dac.device)
                logger.info(f"DAC input tensor shape: {dac_input_tensor.shape}, device: {dac_input_tensor.device}")
                audio_output = self.dac.decode(dac_input_tensor)
                audio_output = audio_output.squeeze(0).cpu() # Remove batch dim, move to CPU
            logger.info(f"Audio generated by DAC, shape: {audio_output.shape}")
        except Exception as e:
            logger.error(f"Error during DAC decoding: {e}")
            logger.error(f"Problematic output_codes (first 5 of each): c1: {c1[:5]}, c2: {c2[:5]}")
            return False

        try:
            sf.write(output_file, audio_output, samplerate=sample_rate)
            logger.info(f"Speech successfully generated and saved to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving audio file {output_file}: {e}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate speech using fine-tuned OuteTTS (v3 interface)")
    parser.add_argument("--config", type=str, help="Path to the config YAML file")
    parser.add_argument("--model_path", type=str, help="Path to the fine-tuned LoRA adapter (or full model if --no_lora)")
    parser.add_argument("--text", type=str, help="Text to convert to speech")
    parser.add_argument("--output_file", type=str, help="Path to save the generated audio")
    parser.add_argument("--no_lora", action="store_true", help="Set if model_path is a full model, not a LoRA adapter")
    parser.add_argument("--base_model", type=str, default="OuteAI/Llama-OuteTTS-1.0-1B", help="Base model name")
    parser.add_argument("--speaker", type=str, default="default_speaker", help="Speaker name/tag used during training")
    parser.add_argument("--lang", type=str, default="ar", help="Language code (e.g., ar, en)")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Max sequence length for model loading")
    
    args = parser.parse_args()
    
    # Load config from file if provided, otherwise use command-line args
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        # Override with command-line args if provided
        config["model_path"] = args.model_path if args.model_path else config.get("model_path")
        config["text"] = args.text if args.text else config.get("text")
        config["output_file"] = args.output_file if args.output_file else config.get("output_file")
        config["use_lora"] = not args.no_lora if args.no_lora is not None else config.get("use_lora", True) # Handle store_true
        config["base_model"] = args.base_model if args.base_model else config.get("base_model", "OuteAI/Llama-OuteTTS-1.0-1B")
        config["speaker"] = args.speaker if args.speaker else config.get("speaker", "default_speaker")
        config["lang"] = args.lang if args.lang else config.get("lang", "ar")
        config["max_seq_length"] = args.max_seq_len if args.max_seq_len else config.get("max_seq_length", 2048)

    else:
        if not args.model_path or not args.text or not args.output_file:
            parser.error("If --config is not used, then --model_path, --text, and --output_file must be provided.")
        
        config = {
            "model_path": args.model_path,
            "text": args.text,
            "output_file": args.output_file,
            "use_lora": not args.no_lora,
            "base_model": args.base_model,
            "speaker": args.speaker,
            "lang": args.lang,
            "max_seq_length": args.max_seq_len,
            # Default generation values
            "top_p": 0.9,
            "temperature": 0.4, # Changed from 0.7 to 0.4 to match notebook
            "repetition_penalty": 1.1, # Changed from 1.2 to 1.1
            "top_k": 40, # Added top_k parameter
            "min_p": 0.05, # Added min_p parameter
            "max_new_tokens": 1024, # Adjusted for DAC tokens, was 2048
            "sample_rate": 24000, # This should ideally come from DAC's sample_rate
        }
    
    logger.info(f"Effective configuration: {config}")

    # Initialize generator
    generator = OuteTTSGeneratorV3(
        model_path=config["model_path"],
        use_lora=config.get("use_lora", True),
        base_model_name=config.get("base_model", "OuteAI/Llama-OuteTTS-1.0-1B"),
        max_seq_length=config.get("max_seq_length", 2048)
    )
    
    # Generate speech
    success = generator.generate_speech(
        text=config["text"],
        output_file=config["output_file"],
        speaker_name=config.get("speaker", "default_speaker"),
        lang=config.get("lang", "ar"),
        top_p=config.get("top_p", 0.9),
        top_k=config.get("top_k", 40), # Added top_k parameter
        temperature=config.get("temperature", 0.4), # Changed from 0.7 to 0.4 to match notebook
        repetition_penalty=config.get("repetition_penalty", 1.1), # Changed from 1.2 to 1.1
        min_p=config.get("min_p", 0.05), # Added min_p parameter
        max_new_tokens=config.get("max_new_tokens", 1024), # Max DAC tokens
        sample_rate=config.get("sample_rate", 24000) # ideally dac.sample_rate
    )

    if success:
        logger.info(f"Speech generated successfully: {config['output_file']}")
    else:
        logger.error("Speech generation failed.")

if __name__ == "__main__":
    main() 