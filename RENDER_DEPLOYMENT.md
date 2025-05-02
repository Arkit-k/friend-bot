# Deploying Your Discord Bot to Render

This guide will walk you through deploying your Friendship Bot to Render's free tier.

## Prerequisites

1. A [Render](https://render.com/) account
2. Your Discord bot token
3. Other API keys (Gemini, Reddit, etc.) if you're using those features

## Step 1: Prepare Your Repository

If your code is not already in a GitHub repository, you'll need to create one:

1. Create a new repository on GitHub
2. Push your code to the repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git push -u origin main
   ```

## Step 2: Deploy to Render

### Option 1: Using the Render Dashboard (Easiest)

1. Log in to your [Render Dashboard](https://dashboard.render.com/)
2. Click on "New" and select "Web Service"
3. Connect your GitHub repository
4. Configure your service:
   - **Name**: `friendship-bot` (or any name you prefer)
   - **Environment**: `Python`
   - **Region**: Choose the region closest to you
   - **Branch**: `main` (or your default branch)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python discord_bot.py`
   - **Plan**: Free

5. Add your environment variables:
   - Click on "Advanced" and then "Add Environment Variable"
   - Add each of these variables:
     - `DISCORD_TOKEN`: Your Discord bot token
     - `GEMINI_API_KEY`: Your Gemini API key (if using)
     - `REDDIT_CLIENT_ID`: Your Reddit client ID (if using)
     - `REDDIT_CLIENT_SECRET`: Your Reddit client secret (if using)
     - `REDDIT_USER_AGENT`: Your Reddit user agent (if using)

6. Click "Create Web Service"

### Option 2: Using render.yaml (More Advanced)

1. Make sure you have the `render.yaml` file in your repository
2. Log in to your [Render Dashboard](https://dashboard.render.com/)
3. Click on "New" and select "Blueprint"
4. Connect your GitHub repository
5. Render will detect the `render.yaml` file and create the services defined in it
6. You'll need to manually add your environment variables after deployment

## Step 3: Set Environment Variables

After deployment, you need to set your environment variables:

1. Go to your service in the Render Dashboard
2. Click on "Environment" in the left sidebar
3. Add each of these variables:
   - `DISCORD_TOKEN`: Your Discord bot token
   - `GEMINI_API_KEY`: Your Gemini API key (if using)
   - `REDDIT_CLIENT_ID`: Your Reddit client ID (if using)
   - `REDDIT_CLIENT_SECRET`: Your Reddit client secret (if using)
   - `REDDIT_USER_AGENT`: Your Reddit user agent (if using)
4. Click "Save Changes"

## Step 4: Verify Deployment

1. Go to the "Logs" tab in your Render Dashboard
2. Check that your bot is running without errors
3. Verify that your bot is online in Discord

## Important Notes

1. **Free Tier Limitations**: Render's free tier has some limitations:
   - Services on the free plan will spin down after 15 minutes of inactivity
   - They will spin back up when a new request comes in
   - This means your bot might have a delay when responding after periods of inactivity

2. **Keeping Your Bot Active**: To keep your bot active, you can:
   - Set up a health check endpoint in your bot
   - Use a service like [UptimeRobot](https://uptimerobot.com/) to ping your bot regularly

3. **Costs**: The configuration in `render.yaml` uses the free plan, so there should be no costs. However, always monitor your usage to avoid unexpected charges.

## Troubleshooting

If your bot isn't working after deployment, check these common issues:

1. **Privileged Intents**: Make sure you've enabled privileged intents in the Discord Developer Portal
2. **Environment Variables**: Verify all environment variables are set correctly
3. **Logs**: Check the logs in the Render Dashboard for any errors
4. **Discord Status**: Verify your bot appears online in Discord

If you're still having issues, you can check the Render logs for more detailed error messages.
