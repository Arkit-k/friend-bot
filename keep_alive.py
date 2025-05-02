"""
Simple web server to keep the bot alive on hosting platforms like Render.
This creates a basic HTTP server that responds to health checks.
"""

from flask import Flask
from threading import Thread

app = Flask('')

@app.route('/')
def home():
    """Return a simple health check response."""
    return "Bot is alive!"

def run():
    """Run the Flask app."""
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    """Start the web server in a separate thread."""
    t = Thread(target=run)
    t.daemon = True
    t.start()
    print("Keep alive server running at http://0.0.0.0:8080")
