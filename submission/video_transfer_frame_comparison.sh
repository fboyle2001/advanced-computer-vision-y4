#!/bin/bash
# Use as video_transfer_frame_comparison.sh <original video> <cyclegan video> <recyclegan video>

mkdir -p ./video_transfer/frames/original
mkdir -p ./video_transfer/frames/cyclegan
mkdir -p ./video_transfer/frames/recyclegan

ffmpeg -i $1 -vf yadif -qscale:v 2 ./video_transfer/frames/original/%05d.jpg
ffmpeg -i $2 -vf yadif -qscale:v 2 ./video_transfer/frames/cyclegan/%05d.jpg
ffmpeg -i $3 -vf yadif -qscale:v 2 ./video_transfer/frames/recyclegan/%05d.jpg