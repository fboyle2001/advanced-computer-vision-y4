#!/bin/bash
# Use as frame_extraction.sh <top-level data directory>
# sh frame_extraction.sh ./data

mkdir -p ./extracted_data/frames/Train/Game
mkdir -p ./extracted_data/frames/Train/Movie
mkdir -p ./extracted_data/frames/Test

for file in "$1"/Train/Game/*; do
  basefile=Train_Game_$(basename "$file" .mp4)
  ffmpeg -i "$file" -vf yadif -qscale:v 2 ./extracted_data/frames/Train/Game/"$basefile"_%05d.jpg
done

for file in "$1"/Train/Movie/*; do
  basefile=Train_Movie_$(basename "$file" .mp4)
  ffmpeg -i "$file" -vf yadif -qscale:v 2 ./extracted_data/frames/Train/Movie/"$basefile"_%05d.jpg
done

for file in "$1"/Test/*; do
  basefile=Test_$(basename "$file" .mp4)
  ffmpeg -i "$file" -vf yadif -qscale:v 2 ./extracted_data/frames/Test/"$basefile"_%05d.jpg
done