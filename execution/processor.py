from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from core.transform_engine import TransformEngine

@dataclass
class ExecutionConfig:
    model_name: str
    num_teaching_shots: int
    num_practice_shots: int
    learning_rate: float
    max_iterations: int = 100
    convergence_threshold: float = 0.01

class Executor:
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.transform_engine = None
        
    def initialize_transformation(self, transformation_type: str):
        self.transform_engine = TransformEngine.create(
            transformation_type,
            self.config.model_name
        )
        
    def run_execution(self, target_query: str) -> Dict:
        if self.transform_engine is None:
            raise ValueError("Transform engine not initialized")
            
        # Run teaching phase to generate and test ciphers
        teaching_results = self._run_teaching_phase()
        
        # Use the best cipher from teaching phase
        best_result = max(teaching_results, key=lambda x: x['success_score'])
        self.transform_engine.cypher = best_result['cypher']
        
        # Transform the target query
        result = self.transform_engine.transform_query(target_query)
        
        return {
            "query": target_query,
            "transformed": result['transformed'],
            "success_score": result['success_score']
        }
        
    def _run_teaching_phase(self) -> List[Dict]:
        results = []
        for _ in range(self.config.num_teaching_shots):
            result = self.transform_engine.generate_teaching_example()
            results.append(result)
        return results
