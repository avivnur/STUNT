#!/bin/bash

# Check if the input file exists
if [ ! -f "$1" ]; then
    echo "File not found!"
    exit 1
fi

# Calculate the average of the test scores
awk '{ sum += $4 } END { print "Average:", sum / NR }' "$1"
