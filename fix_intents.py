"""
Script to fix Discord bot intents issues.

This script modifies the discord_bot.py file to handle privileged intents more gracefully.
"""

import os
import re
import sys

def fix_intents():
    """Fix the intents in discord_bot.py."""
    # Path to the Discord bot file
    bot_file = "discord_bot.py"
    
    # Check if the file exists
    if not os.path.exists(bot_file):
        print(f"Error: {bot_file} not found")
        return False
    
    # Read the file
    with open(bot_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the intents section
    intents_pattern = r"# Bot configuration\s+intents = discord\.Intents\.default\(\)\s+intents\.message_content = True\s+intents\.members = True\s+\s+bot = commands\.Bot\(command_prefix='!', intents=intents\)"
    
    # Replace with the new intents code
    new_intents_code = """# Bot configuration
try:
    # Try with privileged intents first
    intents = discord.Intents.default()
    intents.message_content = True  # This is a privileged intent
    intents.members = True  # This is a privileged intent
    print("Using privileged intents (message_content and members)")
except Exception as e:
    # Fall back to default intents if there's an error
    print(f"Error setting up privileged intents: {e}")
    print("Falling back to default intents")
    intents = discord.Intents.default()

bot = commands.Bot(command_prefix='!', intents=intents)"""
    
    # Check if the pattern matches
    if re.search(intents_pattern, content):
        # Replace the intents section
        new_content = re.sub(intents_pattern, new_intents_code, content)
        
        # Write the modified file
        with open(bot_file, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        print(f"Successfully updated {bot_file} with improved intents handling")
        return True
    else:
        # If the pattern doesn't match, try a simpler approach
        intents_section = "# Bot configuration\nintents = discord.Intents.default()\nintents.message_content = True\nintents.members = True\n\nbot = commands.Bot(command_prefix='!', intents=intents)"
        
        if intents_section in content:
            # Replace the intents section
            new_content = content.replace(intents_section, new_intents_code)
            
            # Write the modified file
            with open(bot_file, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            print(f"Successfully updated {bot_file} with improved intents handling")
            return True
        else:
            print("Could not find the intents section in the file")
            print("Please manually update the intents section in discord_bot.py")
            return False

def main():
    """Main function."""
    print("Fixing Discord bot intents...")
    success = fix_intents()
    
    if success:
        print("\nIntents fixed successfully!")
        print("\nTo enable privileged intents in the Discord Developer Portal:")
        print("1. Go to https://discord.com/developers/applications")
        print("2. Select your bot application")
        print("3. Click on 'Bot' in the left sidebar")
        print("4. Scroll down to 'Privileged Gateway Intents'")
        print("5. Enable 'MESSAGE CONTENT INTENT' and 'SERVER MEMBERS INTENT'")
        print("6. Click 'Save Changes'")
        print("7. Restart your bot")
    else:
        print("\nFailed to fix intents automatically.")
        print("Please manually update the intents section in discord_bot.py")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
