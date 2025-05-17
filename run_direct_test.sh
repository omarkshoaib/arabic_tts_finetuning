#!/bin/bash

# Enable error reporting
set -e

echo "===== RUNNING DIRECT TEST SCRIPT ====="
echo "This script runs a direct test based on the notebook example pattern"

# Run the direct test script
python3 src/inference/direct_test.py

# Check if the output file was created
if [ -f "direct_test_output.wav" ]; then
    echo "Test successful! Audio file created at direct_test_output.wav"
else
    echo "Test failed! No audio file was created."
fi

echo "===== DIRECT TEST COMPLETE =====" 