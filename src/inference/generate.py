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
        self.use_lora = use_lora
        
        logger.info(f"Initializing OuteTTS v3 generator.")
        logger.info(f"Using LoRA: {use_lora}")
        logger.info(f"Model/Adapter path: {model_path}")
        logger.info(f"Base model: {base_model_name}")
        logger.info(f"Device: {device}")

        # Load tokenizer (usually from the base model)
        logger.info(f"Loading tokenizer from {base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Load base model using Unsloth FastModel
        logger.info(f"Loading base model '{base_model_name}' with Unsloth...")
        self.model, _ = FastModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            dtype=torch.float32,
            load_in_4bit=False, # OuteTTS requires no 4-bit
            device_map=self.device, # Map model to device
        )
        
        # Load LoRA adapter if necessary
        if use_lora:
            logger.info(f"Loading and merging LoRA adapter from: {model_path}")
            try:
                self.model = PeftModel.from_pretrained(self.model, model_path)
                # If you want to merge LoRA weights for faster inference (optional, increases memory)
                # self.model = self.model.merge_and_unload() 
                logger.info("LoRA adapter loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading LoRA adapter: {e}. Check if model_path is correct and contains adapter_model.bin/safetensors.")
                raise
        else:
            # If not using LoRA, model_path should be a full fine-tuned model
            # For now, we assume Unsloth's from_pretrained handled this if model_path = base_model_name
            # If model_path is a different full model, this part might need adjustment or rely on HF from_pretrained
            logger.info(f"Using model from {model_path} without LoRA.")


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
        speaker_name: str = "default_speaker", # Default speaker tag
        lang: str = "ar", # Explicitly using "lang" as per v3
        top_p: float = 0.9,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 2048, # Max tokens for the *entire* output (prompt + generation)
        sample_rate: int = 24000,
    ):
        """
        Generate speech from text using OuteTTS v3 interface.
        """
        # For OuteTTS v1.0, the prompt format is different:
        # format from Oute_TTS_(1B).ipynb sample code
        formatted_text = f"<|text_start|>{text}<|text_end|>"
        prompt_text_part = "\n".join([
            "<|im_start|>",
            formatted_text,
            "<|audio_start|><|global_features_start|>",
        ])
        
        logger.info(f"Processed text for prompt: {prompt_text_part[:150]}...")
        
        inputs = self.tokenizer(prompt_text_part, return_tensors="pt").to(self.device)
        input_ids_length = inputs.input_ids.shape[1]
        logger.info(f"Input prompt length in tokens: {input_ids_length}")

        logger.info("Generating DAC codes...")
        with torch.no_grad():
            # Generate, ensuring pad_token_id is eos_token_id if not set, or tokenizer.pad_token_id
            # OuteTTS models often use eos_token_id for padding during generation.
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
            
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens, # Max tokens for the DAC codes part
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_token_id, # Important for generation
                eos_token_id=self.tokenizer.eos_token_id, # Ensure generation stops
            )
        
        # Extract generated token IDs, excluding the input prompt
        # generated_token_ids will be a list of integers.
        generated_token_ids = outputs[0, input_ids_length:].tolist()
        logger.info(f"Raw generated token IDs (after prompt): {len(generated_token_ids)}")

        # Convert to string for regex processing
        logger.info("Converting generated token IDs to audio codes...")
        decoded_output = self.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        
        # Log a portion of the decoded output for debugging
        logger.info(f"Decoded output (first 200 chars): {decoded_output[:200]}")
        
        # OuteTTS v1.0 uses c1_XXX and c2_XXX format instead of DAC_XXX
        import re
        c1 = list(map(int, re.findall(r"<\|c1_(\d+)\|>", decoded_output)))
        c2 = list(map(int, re.findall(r"<\|c2_(\d+)\|>", decoded_output)))
        
        logger.info(f"Found {len(c1)} c1 tokens and {len(c2)} c2 tokens")
        
        # Ensure equal number of tokens from both codebooks
        t = min(len(c1), len(c2))
        if t == 0:
            logger.error("No valid audio tokens found in the output")
            return False
            
        c1 = c1[:t]
        c2 = c2[:t]
        
        # Format for DAC
        audio_codes = [c1, c2]
        logger.info(f"Prepared audio codes shape: {len(audio_codes)} codebooks, {len(audio_codes[0])} tokens each")
        
        # Create tensor in the format expected by DAC
        try:
            codes_tensor = torch.tensor([audio_codes], dtype=torch.int64).to(self.device)
            logger.info(f"Codes tensor shape: {codes_tensor.shape}")
        except Exception as e:
            logger.error(f"Error creating codes tensor: {e}")
            return False

        # Decode using DAC
        logger.info("Decoding audio codes to audio...")
        try:
            audio_waveform = self.dac.decode(codes_tensor)
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            logger.error(f"Codes tensor shape: {codes_tensor.shape}, dtype: {codes_tensor.dtype}")
            return False

        if audio_waveform is None or audio_waveform.ndim == 0 or audio_waveform.shape[-1] == 0:
            logger.error("DAC decoding resulted in empty or invalid audio.")
            return False

        # Ensure audio is 1D numpy array for soundfile
        audio_numpy = audio_waveform.squeeze().cpu().numpy()
        if audio_numpy.ndim > 1: # If still 2D (e.g. [1, L]), squeeze further if L=1, or take first channel.
            audio_numpy = audio_numpy.squeeze()
            if audio_numpy.ndim > 1 and audio_numpy.shape[0] == 1 : # handles case like [1,L]
                audio_numpy = audio_numpy[0]
            elif audio_numpy.ndim > 1 :
                logger.warning(f"Decoded audio is multi-channel ({audio_numpy.shape}), taking first channel.")
                audio_numpy = audio_numpy[0]


        logger.info(f"Saving audio to {output_file} (Sample Rate: {sample_rate})...")
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        sf.write(output_file, audio_numpy, sample_rate) # DAC output SR might differ, use dac.sample_rate
        logger.info("Speech generation complete!")
        return True

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
            "temperature": 0.7,
            "repetition_penalty": 1.2,
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
        temperature=config.get("temperature", 0.7),
        repetition_penalty=config.get("repetition_penalty", 1.2),
        max_new_tokens=config.get("max_new_tokens", 1024), # Max DAC tokens
        sample_rate=config.get("sample_rate", 24000) # ideally dac.sample_rate
    )

    if success:
        logger.info(f"Speech generated successfully: {config['output_file']}")
    else:
        logger.error("Speech generation failed.")

if __name__ == "__main__":
    main() 