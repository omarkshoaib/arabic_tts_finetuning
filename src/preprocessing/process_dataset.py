#!/usr/bin/env python3
"""
Wrapper script to process the dataset for OuteTTS v1.0 fine-tuning.
"""

import os
import sys
import argparse
from loguru import logger
from data_creator import process_csv_metadata

def main():
    """Main function to process the dataset."""
    parser = argparse.ArgumentParser(description="Process the dataset for OuteTTS v1.0 fine-tuning")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing the data (should have 'wavs' subdirectory and metadata CSV)")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the processed data")
    parser.add_argument("--metadata_file", type=str, default="metadata_train.csv", 
                        help="Name of the metadata CSV file within data_dir")
    parser.add_argument("--model_tokenizer", type=str, default="OuteAI/Llama-OuteTTS-1.0-1B", 
                        help="Path to the model tokenizer")
    parser.add_argument("--whisper_model", type=str, default="turbo", 
                        help="Whisper model to use (tiny, base, small, medium, large, turbo)")
    parser.add_argument("--batch_size", type=int, default=1000, 
                        help="Batch size for saving data")
    
    args = parser.parse_args()
    
    # Check if the data directory exists
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Check if the wavs directory exists
    wavs_dir = os.path.join(args.data_dir, "wavs")
    if not os.path.exists(wavs_dir):
        logger.error(f"Wavs directory not found: {wavs_dir}")
        sys.exit(1)
    
    # Check if the metadata file exists
    metadata_path = os.path.join(args.data_dir, args.metadata_file)
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        sys.exit(1)
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get device information
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Processing dataset from {args.data_dir}")
    logger.info(f"Using device: {device}")
    logger.info(f"Using Whisper model: {args.whisper_model}")
    
    # Process the metadata
    process_csv_metadata(
        csv_path=metadata_path,
        wavs_dir=wavs_dir,
        output_path=args.output_dir,
        model_tokenizer_path=args.model_tokenizer,
        whisper_model=args.whisper_model,
        device=device,
        batch_size=args.batch_size
    )
    
    logger.info(f"Dataset processing complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 