@echo off
echo Starting Friendship Bot...

echo Checking for model files...
if not exist friendship_model.h5 (
    echo Model files not found. Creating dummy model files...
    python create_dummy_model.py
) else (
    echo Model files found.
)

echo Starting the bot...
python discord_bot.py

pause
