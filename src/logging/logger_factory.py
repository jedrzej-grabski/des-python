from typing import Type, Dict
from src.logging.base_logger import BaseLogger
from src.logging.des_logger import DESLogger


class LoggerFactory:
    """Factory for creating algorithm-specific loggers."""

    _loggers: Dict[str, Type[BaseLogger]] = {}

    @classmethod
    def register_logger(cls, algorithm_name: str, logger_class: Type[BaseLogger]):
        """Register a logger for an algorithm."""
        cls._loggers[algorithm_name] = logger_class

    @classmethod
    def create_logger(cls, algorithm_name: str, config) -> BaseLogger:
        """Create a logger for the specified algorithm."""
        if algorithm_name in cls._loggers:
            return cls._loggers[algorithm_name](config)
        else:
            raise NotImplementedError("")


# Register known loggers
LoggerFactory.register_logger("DES", DESLogger)
