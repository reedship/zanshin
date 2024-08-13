#!/bin/bash

# Directory containing the .mov files
directory="/Volumes/trainingdata/"

# Find all .mov files (case-insensitive)
mov_files=$(find "$directory" -type f -iname "*.mov")
count=$(echo "$mov_files" | wc -l)

# Print the count of found .mov files
echo "Found $count .mov files."

# Convert each .mov file to .mp4
i=0
for mov_file in $mov_files; do
  i=$((i + 1))
  echo "Processing file $i of $count: $mov_file"

  # Construct the output .mp4 file name
  mp4_file="${mov_file%.*}.mp4"

  # Convert the .mov file to .mp4
  ffmpeg -i "$mov_file" -vcodec h264 -acodec mp2 "$mp4_file"
done

echo "Conversion completed."
