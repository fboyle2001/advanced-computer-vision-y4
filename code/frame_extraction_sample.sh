#!/bin/bash
# Use as frame_extraction_sample.sh <top-level data directory> <frames per second>
# sh frame_extraction_sample.sh ./data 8

mkdir -p ./frames/Train/Game
mkdir -p ./frames/Train/Movie
mkdir -p ./frames/Test

for file in "$1"/Train/Game/*; do
  basefile=Train_Game_$(basename "$file" .mp4)
  ffmpeg -i "$file" -vf yadif -vf fps="$2" -qscale:v 2 ./frames/Train/Game/"$basefile"_%05d.jpg
done

for file in "$1"/Train/Movie/*; do
  basefile=Train_Movie_$(basename "$file" .mp4)
  ffmpeg -i "$file" -vf yadif -vf fps="$2" -qscale:v 2 ./frames/Train/Movie/"$basefile"_%05d.jpg
done

for file in "$1"/Test/*; do
  basefile=Test_$(basename "$file" .mp4)
  ffmpeg -i "$file" -vf yadif -vf fps="$2" -qscale:v 2 ./frames/Test/"$basefile"_%05d.jpg
done