#!/usr/bin/env python3
"""
Inference script for generating speech using the fine-tuned OuteTTS model.
"""

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
from outetts.models.llama_tts import LlamaTTS
from outetts.processor.text import OuteTTSTextProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OuteTTSGenerator:
    def __init__(
        self,
        model_path: str,
        use_lora: bool = True,
        base_model_name: str = "OuteAI/Llama-OuteTTS-1.0-1B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dac_ckpt: Optional[str] = None,
    ):
        """
        Initialize the OuteTTS generator.
        
        Args:
            model_path: Path to the fine-tuned model or LoRA adapter
            use_lora: Whether the model_path points to a LoRA adapter
            base_model_name: Name of the base model to use with LoRA
            device: Device to run inference on ("cuda" or "cpu")
            dac_ckpt: Path to DAC checkpoint (if not using the default one)
        """
        self.device = device
        self.use_lora = use_lora
        
        logger.info(f"Initializing OuteTTS generator with model: {model_path}")
        logger.info(f"Device: {device}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        model_name = base_model_name if use_lora else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load base model
        logger.info(f"Loading base model: {model_name}")
        self.model = LlamaTTS.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Must use fp32 for OuteTTS
            device_map=self.device
        )
        
        # Load LoRA adapter if necessary
        if use_lora:
            logger.info(f"Loading LoRA adapter from: {model_path}")
            self.model = PeftModel.from_pretrained(self.model, model_path)
        
        # Initialize DAC
        logger.info("Initializing DAC interface...")
        self.dac = DacInterface(
            dac_ckpt=dac_ckpt,
            device=self.device,
        )
        
        # Initialize text processor
        logger.info("Initializing text processor...")
        self.text_processor = OuteTTSTextProcessor()
        
    def generate_speech(
        self,
        text: str,
        output_file: str,
        speaker_name: str = "gpt",
        language: str = "ar",
        top_p: float = 0.9,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 2048,
        sample_rate: int = 24000,
    ):
        """
        Generate speech from text.
        
        Args:
            text: Text to convert to speech
            output_file: Path to save the generated audio
            speaker_name: Name of the speaker
            language: Language code (default: "ar" for Arabic)
            top_p: Top-p sampling parameter
            temperature: Temperature for sampling
            repetition_penalty: Repetition penalty parameter
            max_new_tokens: Maximum number of new tokens to generate
            sample_rate: Sample rate of the output audio
        """
        # Process text
        logger.info(f"Processing text: {text[:50]}...")
        processed_text = self.text_processor.process(text, language=language)
        
        # Create prompt
        prompt = f"<|lang:{language}|>[{speaker_name}]:{processed_text}<|startmedia|><|dac|>"
        logger.info(f"Created prompt: {prompt[:100]}...")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        logger.info("Generating speech...")
        with torch.no_grad():
            outputs: CausalLMOutputWithPast = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Get the predicted DAC codes
        predicted_ids = outputs.sequences[0].tolist()
        input_length = inputs.input_ids.shape[1]
        dac_code_string = self.tokenizer.decode(predicted_ids[input_length:])
        
        # Extract DAC codes
        logger.info("Extracting DAC codes...")
        try:
            dac_codes = self.extract_dac_codes(dac_code_string)
            
            # Convert DAC codes to audio
            logger.info("Converting DAC codes to audio...")
            audio = self.dac.decode(dac_codes)
            
            # Save audio
            logger.info(f"Saving audio to {output_file}...")
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            sf.write(output_file, audio, sample_rate)
            logger.info("Done!")
            return True
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return False
    
    def extract_dac_codes(self, dac_code_string: str):
        """
        Extract DAC codes from the generated string.
        
        Args:
            dac_code_string: String containing DAC codes
            
        Returns:
            numpy.ndarray: Array of DAC codes
        """
        # Remove any text before the first digit
        clean_text = ""
        started = False
        
        for char in dac_code_string:
            if char.isdigit() or char == "-" or char == ",":
                started = True
            
            if started:
                clean_text += char
        
        # Split by commas and convert to integers
        try:
            # First try comma-separated format
            codes = np.array([int(code.strip()) for code in clean_text.split(",") if code.strip()])
        except ValueError:
            # If that fails, try space-separated format
            codes = np.array([int(code.strip()) for code in clean_text.split() if code.strip()])
        
        return codes

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate speech using fine-tuned OuteTTS")
    parser.add_argument("--config", type=str, help="Path to the config YAML file")
    parser.add_argument("--model_path", type=str, help="Path to the fine-tuned model or LoRA adapter")
    parser.add_argument("--text", type=str, help="Text to convert to speech")
    parser.add_argument("--output_file", type=str, help="Path to save the generated audio")
    parser.add_argument("--use_lora", action="store_true", help="Whether the model_path points to a LoRA adapter")
    parser.add_argument("--base_model", type=str, default="OuteAI/Llama-OuteTTS-1.0-1B", help="Base model for LoRA")
    parser.add_argument("--speaker", type=str, default="gpt", help="Speaker name")
    parser.add_argument("--language", type=str, default="ar", help="Language code")
    
    args = parser.parse_args()
    
    # Load config from file if provided, otherwise use command-line args
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        # Override config with command-line args if provided
        if args.model_path:
            config["model_path"] = args.model_path
        if args.text:
            config["text"] = args.text
        if args.output_file:
            config["output_file"] = args.output_file
        if args.use_lora:
            config["use_lora"] = args.use_lora
        if args.base_model:
            config["base_model"] = args.base_model
        if args.speaker:
            config["speaker"] = args.speaker
        if args.language:
            config["language"] = args.language
    else:
        if not args.model_path or not args.text or not args.output_file:
            parser.error("Either --config or all of --model_path, --text, and --output_file must be provided")
        
        config = {
            "model_path": args.model_path,
            "text": args.text,
            "output_file": args.output_file,
            "use_lora": args.use_lora,
            "base_model": args.base_model,
            "speaker": args.speaker,
            "language": args.language,
            # Default values
            "top_p": 0.9,
            "temperature": 0.7,
            "repetition_penalty": 1.2,
            "max_new_tokens": 2048,
            "sample_rate": 24000,
        }
    
    # Initialize generator
    generator = OuteTTSGenerator(
        model_path=config["model_path"],
        use_lora=config.get("use_lora", True),
        base_model_name=config.get("base_model", "OuteAI/Llama-OuteTTS-1.0-1B"),
    )
    
    # Generate speech
    generator.generate_speech(
        text=config["text"],
        output_file=config["output_file"],
        speaker_name=config.get("speaker", "gpt"),
        language=config.get("language", "ar"),
        top_p=config.get("top_p", 0.9),
        temperature=config.get("temperature", 0.7),
        repetition_penalty=config.get("repetition_penalty", 1.2),
        max_new_tokens=config.get("max_new_tokens", 2048),
        sample_rate=config.get("sample_rate", 24000),
    )

if __name__ == "__main__":
    main() 