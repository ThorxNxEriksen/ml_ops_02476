import sys
from loguru import logger

# Configure loguru to log to both stdout and a file
logger.remove()
logger.add(sys.stdout, level="WARNING")  # Log to console
logger.add("logs.log", level="DEBUG", rotation="100 MB")  # Log to file

# Export logger for use in other scripts
__all__ = ["logger"]