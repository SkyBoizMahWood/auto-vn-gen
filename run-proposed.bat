@echo off

REM Activate the virtual environment
call .venv\Scripts\activate

REM Run the Python script
python main.py generate-story-with ^
    --approach proposed ^
    --game-genre "japanese visual novel"
    --num-chapters 3 ^
    --num-endings 3 ^
    --num-main-characters 10 ^
    --num-main-scenes 10 ^
    --min-num-choices 2 ^
    --max-num-choices 3 ^
    --min-num-choices-opportunity 1 ^
    --max-num-choices-opportunity 3 ^
    --themes "romance" ^
    --themes "comedy" ^
    --themes "drama" ^
    --enable-image-generation

REM python main.py generate-story-with --approach proposed --game-genre "japanese visual novel" --num-chapters 3 --num-endings 3 --num-main-characters 10 --num-main-scenes 10 --min-num-choices 2 --max-num-choices 3 --min-num-choices-opportunity 1 --max-num-choices-opportunity 3 --themes "romance" --themes "comedy" --themes "drama" --enable-image-generation

REM python main.py generate-story-with --approach proposed --game-genre "japanese visual novel" --num-chapters 3 --num-endings 3 --num-main-characters 5 --num-main-scenes 5 --min-num-choices 2 --max-num-choices 3 --min-num-choices-opportunity 1 --max-num-choices-opportunity 3 --themes "romance" --themes "comedy" --themes "drama" --enable-image-generation

REM python main.py generate-story --game-genre "japanese visual novel" --num-chapters 3 --num-endings 3 --num-main-characters 5 --num-main-scenes 5 --min-num-choices 2 --max-num-choices 3 --min-num-choices-opportunity 1 --max-num-choices-opportunity 2 --themes "romance" --themes "comedy" --themes "drama" --enable-image-generation

REM python main.py generate-story --game-genre "japanese visual novel" --num-chapters 2 --num-endings 10 --num-main-characters 10 --num-main-scenes 10 --min-num-choices 3 --max-num-choices 3 --min-num-choices-opportunity 1 --max-num-choices-opportunity 1 --themes "romance" --themes "comedy" --themes "drama"

REM python main.py generate-story --num-chapters 3 --min-num-choices 2 --max-num-choices 3 --min-num-choices-opportunity 1 --max-num-choices-opportunity 1 --themes "romance" --themes "comedy" --themes "drama"

REM python main.py generate-story --num-chapters 3 --min-num-choices 2 --max-num-choices 3 --min-num-choices-opportunity 1 --max-num-choices-opportunity 1 --themes "adventure" --themes "high-fantasy" --themes "science fiction"

REM python main.py generate-story --num-chapters 3 --min-num-choices 2 --max-num-choices 3 --min-num-choices-opportunity 1 --max-num-choices-opportunity 1 --themes "Rewriting Fate" --themes "The Fragility of Memory" --themes "The Weight of Regret"

REM python main.py batch-generation --themes "Romance, Mystery, Fantasy, Sci-fi" --min-num-choices-opportunity 1 --max-num-choices-opportunity 1 --n-stories 5