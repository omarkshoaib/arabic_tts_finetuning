#!/usr/bin/env python3
"""
Cleans a metadata CSV file for OuteTTS fine-tuning by attempting to correctly
identify audio_file, text, and speaker_name from potentially mixed-up columns.
"""
import polars as pl
import argparse
import os
from loguru import logger

# Known speaker identifiers (add more if needed, used to help differentiate text from speaker)
KNOWN_SPEAKERS_LOWERCASE = ["@mariam", "@chatgpt", "nan"]

def is_valid_path(value: str) -> bool:
    """Checks if a value string looks like a valid audio path for this project."""
    if not isinstance(value, str):
        return False
    s = value.strip()
    return s.startswith("wavs/") and s.lower().endswith(".wav")

def is_speaker_tag(value: str) -> bool:
    """Checks if a value string looks like a speaker tag or known speaker."""
    if not isinstance(value, str):
        return False
    s = value.strip().lower()
    return s.startswith("@") or s in KNOWN_SPEAKERS_LOWERCASE

def clean_csv(input_csv_path: str, output_csv_path: str):
    """
    Cleans the input CSV and writes the result to the output CSV.

    Args:
        input_csv_path: Path to the input metadata CSV file.
        output_csv_path: Path to save the cleaned metadata CSV file.
    """
    logger.info(f"Starting CSV cleaning process for: {input_csv_path}")
    try:
        df = pl.read_csv(input_csv_path, separator="|", truncate_ragged_lines=True, has_header=True)
        logger.info(f"Successfully read {len(df)} rows from input CSV.")
    except Exception as e:
        logger.error(f"Failed to read input CSV {input_csv_path}: {e}")
        return

    cleaned_rows = []
    skipped_rows_count = 0

    expected_columns = ['audio_file', 'text', 'speaker_name']
    for col in expected_columns:
        if col not in df.columns:
            logger.error(f"Input CSV is missing expected column: '{col}'. Aborting.")
            # Attempt to show available columns for debugging
            logger.info(f"Available columns: {df.columns}")
            return


    for i, row_dict in enumerate(df.to_dicts()):
        val_audio_col = str(row_dict.get('audio_file', '') or '').strip()
        val_text_col = str(row_dict.get('text', '') or '').strip()
        val_speaker_col = str(row_dict.get('speaker_name', '') or '').strip()

        potential_paths = []
        path_source_columns = {} # To remember where the path came from

        if is_valid_path(val_audio_col):
            potential_paths.append(val_audio_col)
            path_source_columns[val_audio_col] = 'audio_file'
        if is_valid_path(val_text_col):
            potential_paths.append(val_text_col)
            path_source_columns[val_text_col] = 'text'
        if is_valid_path(val_speaker_col):
            potential_paths.append(val_speaker_col)
            path_source_columns[val_speaker_col] = 'speaker_name'
        
        final_path = None
        final_text = ""
        final_speaker = ""

        if len(potential_paths) == 1:
            final_path = potential_paths[0]
            
            non_path_values = []
            if path_source_columns[final_path] != 'audio_file':
                non_path_values.append(val_audio_col)
            if path_source_columns[final_path] != 'text':
                non_path_values.append(val_text_col)
            if path_source_columns[final_path] != 'speaker_name':
                non_path_values.append(val_speaker_col)

            val1 = non_path_values[0] if len(non_path_values) > 0 else ""
            val2 = non_path_values[1] if len(non_path_values) > 1 else ""

            # Determine text and speaker from remaining values
            if is_speaker_tag(val1):
                final_speaker = val1
                final_text = val2
            elif is_speaker_tag(val2):
                final_speaker = val2
                final_text = val1
            else:
                # If neither is a clear speaker tag, assume the longer one is text
                if len(val1) > len(val2):
                    final_text = val1
                    final_speaker = val2 
                elif len(val2) > 0 : # val2 is longer or val1 is empty
                    final_text = val2
                    final_speaker = val1
                else: # Both are empty
                    final_text = val1 # effectively empty
                    final_speaker = val2 # effectively empty


            # Clean up and default speaker
            final_text = final_text.strip()
            final_speaker = final_speaker.strip()

            if not final_speaker or final_speaker.lower() == 'nan':
                final_speaker = 'default_speaker'
            
            if not final_text:
                logger.warning(f"Row {i+1}: Could not determine text for path '{final_path}'. Original values - audio_file: '{val_audio_col}', text: '{val_text_col}', speaker: '{val_speaker_col}'. Skipping.")
                skipped_rows_count +=1
                continue
            
            cleaned_rows.append({
                "audio_file": final_path,
                "text": final_text,
                "speaker_name": final_speaker
            })

        elif len(potential_paths) == 0:
            logger.warning(f"Row {i+1}: No valid audio path found. Original values - audio_file: '{val_audio_col}', text: '{val_text_col}', speaker: '{val_speaker_col}'. Skipping.")
            skipped_rows_count +=1
        else: # len(potential_paths) > 1
            logger.warning(f"Row {i+1}: Multiple valid audio paths found: {potential_paths}. Original values - audio_file: '{val_audio_col}', text: '{val_text_col}', speaker: '{val_speaker_col}'. Skipping.")
            skipped_rows_count +=1

    if cleaned_rows:
        cleaned_df = pl.DataFrame(cleaned_rows)
        try:
            cleaned_df.write_csv(output_csv_path, separator=",")
            logger.info(f"Successfully wrote {len(cleaned_df)} cleaned rows to {output_csv_path}")
            logger.info(f"Skipped {skipped_rows_count} rows during cleaning.")
        except Exception as e:
            logger.error(f"Failed to write cleaned CSV to {output_csv_path}: {e}")
    else:
        logger.warning("No rows were cleaned. Output file will not be created.")
        logger.info(f"Skipped {skipped_rows_count} rows during cleaning.")


def main():
    parser = argparse.ArgumentParser(description="Clean a metadata CSV file for OuteTTS.")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the input (potentially messy) metadata CSV file.")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to save the cleaned metadata CSV file.")
    
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    clean_csv(args.input_csv, args.output_csv)

if __name__ == "__main__":
    main() 