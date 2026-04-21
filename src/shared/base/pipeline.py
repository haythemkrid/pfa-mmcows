from abc import ABC, abstractmethod
from typing import Dict, Any

class BasePipeline(ABC):
    """Abstract Base Class for all ML pipelines in the framework."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def run(self) -> None:
        """Executes the pipeline."""
        pass
