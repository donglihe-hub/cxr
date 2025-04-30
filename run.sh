#!/bin/bash

# Output file
OUTPUT_FILE="cross_validation_results.txt"

# Clear the output file if it exists
> "$OUTPUT_FILE"

# Loop through all YAML files in the cross_validation_config directory
for config_file in cross_validation_config/*.yml; do
    echo "Running configuration: $config_file" | tee -a "$OUTPUT_FILE"
    
    # Run the training command and append output to the file
    python cross_validation.py --config "$config_file" 2>&1 | tee -a "$OUTPUT_FILE"
    
    echo -e "\n\n----------------------------------------\n\n" | tee -a "$OUTPUT_FILE"
done

echo "All configurations completed. Results saved to $OUTPUT_FILE"