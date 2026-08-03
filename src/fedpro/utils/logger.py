"""
Production-grade structured logging for FedHealth.
Provides colorized console output, configurable formatting, and file-based audit logging.
"""

import logging
import sys
from typing import Optional
from pathlib import Path

# ANSI Color Codes for terminal formatting
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
GRAY = "\033[90m"

class ColoredFormatter(logging.Formatter):
    """Custom formatter adding ANSI color codes based on log level."""
    
    LEVEL_COLORS = {
        logging.DEBUG: GRAY,
        logging.INFO: CYAN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD + RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, RESET)
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level_name = f"{color}{record.levelname:<8}{RESET}"
        name = f"{MAGENTA}{record.name:<16}{RESET}"
        message = record.getMessage()
        return f"[{timestamp}] {level_name} | {name} | {message}"

def get_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get or create a configured logger with console and optional file handlers.
    
    Args:
        name: Name of the logger (typically module or component name).
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_file: Optional file path to persist structured logs.
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers if logger was already created
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)
        
        # Optional File Handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
    logger.propagate = False
    return logger
