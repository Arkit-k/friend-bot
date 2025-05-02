# Discord Bot Integration Guide

This guide will help you properly integrate your Discord bot with your server and resolve common issues like the "integration requires code grant" error.

## Enabling Privileged Intents

Your bot uses privileged intents (`message_content` and `members`), which need to be explicitly enabled in the Discord Developer Portal:

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Select your bot application
3. Click on "Bot" in the left sidebar
4. Scroll down to "Privileged Gateway Intents"
5. Enable the following intents:
   - **MESSAGE CONTENT INTENT** (required to read message content)
   - **SERVER MEMBERS INTENT** (required to track members)
6. Click "Save Changes"

![Privileged Intents](https://i.imgur.com/JqKDFSA.png)

## Disabling "Requires OAuth2 Code Grant"

If you're seeing "integration requires code grant" error:

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Select your bot application
3. Click on "Bot" in the left sidebar
4. Scroll down and make sure "Requires OAuth2 Code Grant" is turned **OFF**
5. Click "Save Changes"

![OAuth2 Code Grant](https://i.imgur.com/8JYWGvN.png)

## Creating a Proper Invite Link

To add your bot to a server with the correct permissions:

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Select your bot application
3. Click on "OAuth2" in the left sidebar
4. Click on "URL Generator"
5. Under "SCOPES", select:
   - `bot`
   - `applications.commands`
6. Under "BOT PERMISSIONS", select:
   - "Send Messages"
   - "Read Message History"
   - "View Channels"
   - "Embed Links"
   - "Attach Files"
   - "Use Slash Commands"
   - "Add Reactions"
7. Copy the generated URL at the bottom
8. Open the URL in a new browser tab
9. Select the server you want to add the bot to
10. Click "Authorize"

![OAuth2 URL Generator](https://i.imgur.com/QZFjxVJ.png)

## Running the Bot Safely

We've created several scripts to help you run the bot safely:

### Option 1: Using run_bot_safe.py

This script provides detailed error messages if something goes wrong:

```bash
python run_bot_safe.py
```

### Option 2: Using start_bot.bat

This batch file runs the integration script and starts the bot:

```bash
start_bot.bat
```

### Option 3: Manual Start

If you prefer to start the bot manually:

```bash
python discord_bot.py
```

## Troubleshooting Common Issues

### 1. "Privileged Intents Required" Error

**Error Message:**
```
discord.errors.PrivilegedIntentsRequired: Shard ID None is requesting privileged intents that have not been explicitly enabled in the developer portal.
```

**Solution:**
- Enable privileged intents in the Discord Developer Portal as described above
- Or run `python fix_intents.py` to modify your bot to handle this error gracefully

### 2. "Invalid Token" Error

**Error Message:**
```
discord.errors.LoginFailure: Improper token has been passed.
```

**Solution:**
- Reset your token in the Discord Developer Portal
- Update your `.env` file with the new token
- Run `python test_discord_token.py` to verify your token is valid

### 3. Bot Not Responding to Messages

**Possible Causes:**
- Bot doesn't have permission to read messages
- Bot doesn't have permission to send messages
- Bot is not properly connected to Discord

**Solution:**
- Check the bot's permissions in your server
- Make sure the bot is online (shows as online in the member list)
- Check the console for any error messages

### 4. Bot Not Appearing in Server

**Possible Causes:**
- Bot wasn't properly invited to the server
- Bot invitation didn't include proper permissions

**Solution:**
- Generate a new invite link with the correct permissions
- Make sure you're logged in to the correct Discord account when authorizing

## Testing the Bot

After resolving any integration issues, test your bot with these commands:

- `!friendhelp` - Should show the help message
- `@BotName hello` - Bot should respond to mentions
- Send a direct message to the bot - Bot should respond in DMs

## Need More Help?

If you're still having issues:

1. Check the console output for specific error messages
2. Run `python test_discord_token.py` to verify your token
3. Make sure all required packages are installed: `pip install -r requirements.txt`
4. Check your internet connection
5. Verify that Discord's services are operational
