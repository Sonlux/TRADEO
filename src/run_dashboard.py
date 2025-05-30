import os
import sys
import logging
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import and run the dashboard
    logger.info("Initializing dashboard...")
    from visualization.dashboard import app
    
    if __name__ == "__main__":
        logger.info("Starting Stock Market Analyzer Dashboard...")
        logger.info("Open your browser and navigate to http://127.0.0.1:8050/")
        app.run(debug=True, port=8050)
except Exception as e:
    logger.error(f"Failed to start dashboard: {str(e)}")
    logger.error(f"Stack trace: {traceback.format_exc()}")
    sys.exit(1)