#!/usr/bin/env python3
"""
Data preparation for OuteTTS v1.0 fine-tuning.
This script processes audio files using Whisper and DAC for OuteTTS v1.0 model.
"""

import os
import torch
import tempfile
import polars as pl
from tqdm import tqdm
import soundfile as sf
import numpy as np
import whisper
import io
from datasets import Dataset
from loguru import logger
import argparse

# OuteTTS v3 imports
from outetts.version.v3.audio_processor import AudioProcessor
from outetts.version.v3.prompt_processor import PromptProcessor
from outetts.dac.interface import DacInterface
from outetts.models.config import ModelConfig  # For dummy config
from outetts.utils.preprocessing import text_normalizations

class DataCreatorV3:
    def __init__(
            self,
            model_tokenizer_path: str,
            whisper_model_name: str = "turbo",
            device: str = None
        ):
        """
        Initialize the DataCreatorV3 for OuteTTS v1.0 fine-tuning.
        
        Args:
            model_tokenizer_path: Path to the model tokenizer
            whisper_model_name: Name of the Whisper model to use ("tiny", "base", "small", "medium", "large" or "turbo")
            device: Device to use ("cpu", "cuda")
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Create a dummy ModelConfig mainly for device and paths needed by AudioProcessor/DacInterface
        dummy_config = ModelConfig(
            tokenizer_path=model_tokenizer_path,
            device=self.device,
            audio_codec_path=None  # Let AudioProcessor use default DAC path
        )
        
        self.audio_processor = AudioProcessor(config=dummy_config)
        self.prompt_processor = PromptProcessor(model_tokenizer_path)

        logger.info(f"Loading Whisper model: {whisper_model_name} on {self.device}")
        self.whisper_model = whisper.load_model(whisper_model_name, device=self.device)
        logger.info("Whisper model loaded.")

    def create_speaker_representation(self, audio_bytes: bytes, transcript: str):
        """
        Creates a v3-compatible speaker dictionary using Whisper and AudioProcessor.
        
        Args:
            audio_bytes: Audio bytes
            transcript: Text transcript of the audio
            
        Returns:
            Dictionary containing speaker representation compatible with OuteTTS v1.0
        """
        if not audio_bytes or not transcript:
             logger.warning("Missing audio bytes or transcript in create_speaker_representation.")
             return None

        # Whisper needs a file path, so save bytes to a temporary file
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_audio_file:
                tmp_audio_file.write(audio_bytes)
                tmp_audio_file.flush()  # Ensure data is written

                # 1. Get word timings using Whisper
                whisper_result = self.whisper_model.transcribe(tmp_audio_file.name, word_timestamps=True)
                
                # Use the provided transcript for consistency, but Whisper timings
                normalized_transcript = text_normalizations(transcript)

                words_with_timings = []
                if whisper_result and 'segments' in whisper_result:
                    for segment in whisper_result['segments']:
                        if 'words' in segment:
                            for word_info in segment['words']:
                                # Use original word casing/punctuation from Whisper's output
                                cleaned_word = word_info['word'].strip()
                                if cleaned_word:  # Ignore empty strings
                                    words_with_timings.append({
                                        'word': cleaned_word,
                                        'start': float(word_info['start']),
                                        'end': float(word_info['end'])
                                    })
                else:
                    logger.warning(f"Whisper did not return segments/words for: {transcript[:50]}...")
                    return None  # Indicate failure

                if not words_with_timings:
                    logger.warning(f"No word timings extracted by Whisper for: {transcript[:50]}...")
                    return None

                # Prepare data dict for AudioProcessor
                speaker_data_dict = {
                    "audio": {"bytes": audio_bytes},
                    "text": normalized_transcript,  # Use the potentially normalized transcript
                    "words": words_with_timings
                }

                # 2. Use AudioProcessor to create the speaker representation
                v3_speaker = self.audio_processor.create_speaker_from_dict(speaker_data_dict)
                return v3_speaker

        except Exception as e:
            logger.error(f"Error during speaker creation (Whisper/AudioProcessor): {e}")
            return None  # Indicate failure

    def process_dataset(self, dataset: Dataset):
        """
        Process a Dataset containing audio and text pairs.
        
        Args:
            dataset: HuggingFace Dataset object containing audio and text
            
        Returns:
            List of training prompts
        """
        output_data = []
        processed_count = 0
        skipped_count = 0
        
        logger.info("Starting dataset processing...")
        
        for item in tqdm(dataset, desc="Processing Dataset"):
            try:
                if not item.get('audio') or not item.get('text'):
                    logger.warning("Missing audio or text in dataset item. Skipping.")
                    skipped_count += 1
                    continue
                
                audio_data = item['audio']
                text = item['text']
                
                # Extract audio bytes
                if isinstance(audio_data, dict) and 'bytes' in audio_data:
                    audio_bytes = audio_data['bytes']
                elif isinstance(audio_data, dict) and 'array' in audio_data and 'sampling_rate' in audio_data:
                    # Convert numpy array to bytes
                    with io.BytesIO() as audio_buffer:
                        sf.write(audio_buffer, audio_data['array'], audio_data['sampling_rate'], format='WAV')
                        audio_bytes = audio_buffer.getvalue()
                else:
                    logger.warning(f"Unknown audio format: {type(audio_data)}. Skipping.")
                    skipped_count += 1
                    continue
                
                # Create the speaker representation
                speaker = self.create_speaker_representation(audio_bytes, text)
                
                if speaker is None:
                    logger.warning(f"Failed to create speaker for: {text[:50]}... Skipping.")
                    skipped_count += 1
                    continue
                
                # Generate the training prompt for OuteTTS v1.0
                prompt = self.prompt_processor.get_training_prompt(speaker)
                
                output_data.append({
                    "prompt": prompt
                })
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                skipped_count += 1
        
        logger.info(f"Dataset processing finished. Processed: {processed_count}, Skipped: {skipped_count}")
        
        # If we're on a GPU, free up the Whisper model
        if self.device == "cuda":
            logger.info("Moving Whisper model to CPU")
            self.whisper_model = self.whisper_model.cpu()
            torch.cuda.empty_cache()
        
        return output_data

    def save_processed_data(self, data, output_path, batch_size=1000):
        """
        Save processed data to Parquet files.
        
        Args:
            data: List of dictionaries containing prompts
            output_path: Directory to save the data
            batch_size: Number of items per batch
        """
        os.makedirs(output_path, exist_ok=True)
        
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        
        for i, batch in enumerate(batches):
            file_path = os.path.join(output_path, f"{i:06d}.parquet")
            logger.info(f"Saving batch {i} to {file_path}")
            pl.DataFrame(batch).write_parquet(file_path)
        
        logger.info(f"Saved {len(data)} items in {len(batches)} batches to {output_path}")

def process_csv_metadata(csv_path, wavs_dir, output_path, model_tokenizer_path="OuteAI/Llama-OuteTTS-1.0-1B", 
                       whisper_model="turbo", device=None, batch_size=1000):
    """
    Process a CSV metadata file with audio paths and transcripts.
    
    Args:
        csv_path: Path to the CSV metadata file
        wavs_dir: Directory containing the WAV files
        output_path: Directory to save the processed data
        model_tokenizer_path: Path to the model tokenizer
        whisper_model: Whisper model to use
        device: Device to use ("cpu", "cuda")
        batch_size: Batch size for saving data
    """
    # Read the metadata CSV file using polars
    df = pl.read_csv(csv_path, separator='|')
    
    # Create a dataset from the dataframe
    dataset = []
    for i, row in enumerate(df.iter_rows(named=True)):
        if len(row) < 2:
            logger.warning(f"Skipping row {i+1} due to insufficient columns: {row}")
            continue

        # Swapped based on observed error: transcription seems to be in the first column, filename in the second.
        transcript = row[0].strip()
        audio_file_name = row[1].strip()

        if not audio_file_name.startswith('wavs/'):
            logger.warning(f"Skipping invalid audio file path: {audio_file_name}")
            continue
            
        if not transcript:
            logger.warning(f"Skipping empty transcript for {audio_file_name}")
            continue
            
        audio_path = os.path.join(wavs_dir, audio_file_name.replace('wavs/', ''))
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            continue
            
        # Read the audio file
        try:
            array, sample_rate = sf.read(audio_path)
            if len(array.shape) > 1:
                array = np.mean(array, axis=1)  # Convert stereo to mono
            
            dataset.append({
                'audio': {
                    'array': array,
                    'sampling_rate': sample_rate
                },
                'text': transcript,
                'speaker_name': row['speaker_name']
            })
        except Exception as e:
            logger.error(f"Error reading audio file {audio_path}: {e}")
            continue
    
    # Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_list(dataset)
    
    # Process the dataset
    creator = DataCreatorV3(
        model_tokenizer_path=model_tokenizer_path,
        whisper_model_name=whisper_model,
        device=device
    )
    
    processed_data = creator.process_dataset(hf_dataset)
    
    # Save the processed data
    creator.save_processed_data(processed_data, output_path, batch_size)

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Process audio data for OuteTTS v1.0 fine-tuning")
    parser.add_argument("--metadata", type=str, required=True, help="Path to the metadata CSV file")
    parser.add_argument("--wavs_dir", type=str, required=True, help="Directory containing the WAV files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed data")
    parser.add_argument("--model_tokenizer", type=str, default="OuteAI/Llama-OuteTTS-1.0-1B", 
                        help="Path to the model tokenizer")
    parser.add_argument("--whisper_model", type=str, default="turbo", 
                        help="Whisper model to use (tiny, base, small, medium, large, turbo)")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (cpu, cuda)")
    parser.add_argument("--batch_size", type=int, default=1000, 
                        help="Batch size for saving data")
    
    args = parser.parse_args()
    
    logger.info(f"Processing metadata: {args.metadata}")
    logger.info(f"WAVs directory: {args.wavs_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    process_csv_metadata(
        csv_path=args.metadata,
        wavs_dir=args.wavs_dir,
        output_path=args.output_dir,
        model_tokenizer_path=args.model_tokenizer,
        whisper_model=args.whisper_model,
        device=args.device,
        batch_size=args.batch_size
    )
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main() 