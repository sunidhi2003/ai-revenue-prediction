import logging
from src.logging_config import get_logger

def test_logger_configured():
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO
