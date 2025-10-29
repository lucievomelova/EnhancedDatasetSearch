import logging
import sys


def setup_logger(name: str = 'app', level=logging.INFO):
    # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # logger = logging.getLogger(__name__)

    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        # Avoid adding handlers twice if imported multiple times
        logger.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger