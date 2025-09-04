import logging
import os
from datetime import datetime

# Create log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create logs directory
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Full path to log file
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH, 
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info("Logging has started")
    print(f"Logs will be saved in: {LOG_FILE_PATH}")
