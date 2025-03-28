#!/bin/bash

# Script to run repomix on each directory
BASE_DIR="/home/dislove/文档/SolnAI-agents-1"
OUTPUT_DIR="$BASE_DIR/.dist/repomix-outputs"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get list of directories (excluding hidden ones)
dirs=$(find "$BASE_DIR" -maxdepth 1 -type d | grep -v "/\." | sort)

# Loop through each directory
for dir in $dirs; do
  # Skip the base directory itself
  if [ "$dir" == "$BASE_DIR" ]; then
    continue
  fi
  
  # Get directory name
  dir_name=$(basename "$dir")
  echo "Processing directory: $dir_name"
  
  # Run repomix on the directory
  cd "$dir"
  npx repomix --output "$OUTPUT_DIR/$dir_name.xml"
  
  echo "Completed $dir_name"
  echo "------------------------"
done

echo "All directories processed. Results saved in $OUTPUT_DIR"
