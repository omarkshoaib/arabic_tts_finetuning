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
        # Create inference prompt using PromptProcessor
        # The v3 prompt processor typically takes a dictionary or specific args.
        # For simple text-to-speech, it needs the text, speaker, and lang.
        # It handles normalization and formatting.
        
        # We need to construct the text part of the prompt that the model expects to complete.
        # This usually involves language, speaker, and the input text, followed by generation start tokens.
        # The PromptProcessor.get_inference_prompt might be what we need, or construct manually.
        # Example: prompt_dict = {"text": text, "speaker_name": speaker_name, "lang": lang}
        # inference_prompt_details = self.prompt_processor.get_inference_prompt(prompt_dict)
        # prompt_text_part = inference_prompt_details["prompt_text_part"] 
        # Or more directly:
        prompt_text_part = f"<|lang:{lang}|>[{speaker_name}]:{text}<|startmedia|><|dac|>"
        
        logger.info(f"Processed text for prompt: {prompt_text_part[:150]}...")
        
        inputs = self.tokenizer(prompt_text_part, return_tensors="pt").to(self.device)
        input_ids_length = inputs.input_ids.shape[1]

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
        
        # Extract only the generated tokens (DAC codes)
        generated_token_ids = outputs[0][input_ids_length:]
        
        # Decode the generated tokens to get the DAC code string
        # This part needs to be robust to how OuteTTS v3 formats DAC codes
        # It might output special tokens that need to be handled or stripped.
        
        raw_dac_code_string = self.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        logger.info(f"Raw generated DAC string (first 100 chars): {raw_dac_code_string[:100]}")

        # Extract DAC codes (this method might need adjustment for v3)
        # The v3 interface might provide a more direct way or different format.
        # We need to ensure we're only feeding valid integer codes to dac.decode()
        try:
            # Attempt to extract codes from the string.
            # The string might look like "<|dac|> 123 45 67 <|endoftext|>"
            # Or "<|dac|>123,45,67<|endoftext|>"
            # We need to get the numbers between <|dac|> (or generation start) and <|endoftext|> or other terminators.
            
            # A more robust way for v3 might be to use the prompt_processor if it has a method for this,
            # or to carefully parse based on the known structure of generated DAC tokens.
            # For now, let's try a refined version of the original extraction.

            # Find the start of DAC codes (after the initial prompt part that includes <|dac|>)
            # and end (before <|endoftext|> or other stop tokens).
            
            dac_codes_cleaned = []
            # Assuming DAC tokens are represented as individual tokens by the tokenizer,
            # not necessarily a comma/space separated string of digits after decoding.
            # The model generates TOKEN IDs. The tokenizer.decode turns them into strings.
            # If DAC codes are special tokens (e.g., <DAC_001>, <DAC_002>), we need to map them back to integers.
            # If they are just numbers as strings, then parsing is needed.
            # OuteTTS v1.0 and DAC typically use a vocabulary of DAC tokens (e.g., 0-1023 for Encodec).
            # The model should generate these token IDs directly.
            
            # Let's assume the generated_token_ids ARE the DAC code IDs.
            # We need to check if they are within the DAC's expected range.
            # The prompt_processor or dac interface should clarify the expected format.

            # Placeholder: This is a critical part that needs to align with OuteTTS v3's DAC tokenization.
            # If generated_token_ids are directly usable (e.g. integers from 0-1023 for Encodec)
            # and DacInterface.decode expects a flat list/tensor of these integers.
            
            # Let's assume `generated_token_ids` contains the sequence of DAC code indices.
            # We need to filter out any EOS or PAD tokens that might have been generated if max_new_tokens was reached.
            
            valid_dac_token_ids = []
            for token_id in generated_token_ids.tolist():
                if token_id == self.tokenizer.eos_token_id or token_id == self.tokenizer.pad_token_id:
                    break # Stop if we hit EOS/PAD
                # Check if token_id is a valid DAC code index.
                # This depends on the specific DAC used by OuteTTS v1.0.
                # For now, let's assume all generated non-EOS/PAD tokens are DAC indices.
                # A more robust check would be `0 <= token_id < dac_vocab_size`.
                valid_dac_token_ids.append(token_id)

            if not valid_dac_token_ids:
                logger.error("No valid DAC token IDs were generated after EOS/PAD filtering.")
                return False

            logger.info(f"Number of raw DAC token IDs generated: {len(valid_dac_token_ids)}")
            
            try:
                # Determine the number of codebooks from the loaded DAC model
                n_codebooks = self.dac.model.quantizer.n_codebooks
                logger.info(f"DAC model uses {n_codebooks} codebooks (quantizers).")
            except AttributeError:
                logger.error("Could not determine n_codebooks from self.dac.model.quantizer.n_codebooks. This is critical for reshaping.")
                # As a fallback, for the specific 'ibm-research/DAC.speech.v1.0' (24khz_1.5kbps), n_codebooks is 9.
                # This is risky if the model changes. Ideally, this should always be found.
                # Consider making this a parameter or ensuring the dac model object always has this.
                logger.warning("Attempting to use a default n_codebooks=9 due to AttributeError. This might be incorrect.")
                n_codebooks = 9 

            if n_codebooks <= 0:
                logger.error(f"Invalid n_codebooks found or defaulted: {n_codebooks}. Must be positive.")
                return False

            if len(valid_dac_token_ids) == 0:
                 logger.error("No valid DAC tokens available before reshaping.")
                 return False

            # Ensure the number of tokens is divisible by the number of codebooks
            if len(valid_dac_token_ids) % n_codebooks != 0:
                num_tokens_to_trim = len(valid_dac_token_ids) % n_codebooks
                logger.warning(
                    f"Number of DAC tokens ({len(valid_dac_token_ids)}) is not divisible by "
                    f"n_codebooks ({n_codebooks}). Trimming {num_tokens_to_trim} tokens from the end."
                )
                valid_dac_token_ids = valid_dac_token_ids[:-num_tokens_to_trim]
            
            if not valid_dac_token_ids: # Check if trimming made it empty
                 logger.error("No valid DAC tokens left after attempting to make them divisible by n_codebooks.")
                 return False

            # Reshape to (batch_size, n_codebooks, sequence_length_per_codebook)
            # Batch size is 1 for our inference case.
            dac_codes_tensor = torch.tensor(valid_dac_token_ids, dtype=torch.long, device=self.device)
            dac_codes_tensor = dac_codes_tensor.view(1, n_codebooks, -1) 
            logger.info(f"Reshaped DAC codes tensor to: {dac_codes_tensor.shape}")
            
            # Assuming dac.decode can handle a 2D tensor [1, sequence_length] of DAC indices for now.
            # This is the most likely point of failure if the DAC part is not correctly interfaced.
            
            audio_waveform = self.dac.decode(dac_codes_tensor) # Expects tensor of indices
            
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
            
        except Exception as e:
            logger.error(f"Error during DAC code extraction or audio generation: {e}", exc_info=True)
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