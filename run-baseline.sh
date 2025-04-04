#!/bin/zsh
python main.py generate-story-with \
  --approach baseline \
  --num-chapters 3 \
  --min-num-choices 2 \
  --max-num-choices 3 \
  --min-num-choices-opportunity 1 \
  --max-num-choices-opportunity 2 \
  --enable-image-generation
  --themes "adventure"
  --themes "high-fantasy"
  --themes "science fiction"