from typing import Type
from src.algorithms.choices import AlgorithmChoice
from src.logging.base_logger import BaseLogger
from src.logging.des_logger import DESLogger


class LoggerFactory:
    """Factory for creating algorithm-specific loggers."""

    _loggers: dict[AlgorithmChoice, Type[BaseLogger]] = {}

    @classmethod
    def register_logger(
        cls, algorithm: AlgorithmChoice, logger_class: Type[BaseLogger]
    ):
        """Register a logger for an algorithm."""
        cls._loggers[algorithm] = logger_class

    @classmethod
    def create_logger(cls, algorithm: AlgorithmChoice, config) -> BaseLogger:
        """Create a logger for the specified algorithm."""
        if algorithm in cls._loggers:
            return cls._loggers[algorithm](config)
        else:
            raise NotImplementedError("")


# Register known loggers
LoggerFactory.register_logger(AlgorithmChoice.DES, DESLogger)
