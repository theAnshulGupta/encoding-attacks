from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from ..core.transform_engine import TransformEngine

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
            
        teaching_results = self._run_teaching_phase()
        practice_results = self._run_practice_phase()
        final_result = self._run_target_phase(target_query)
        
        return {
            "teaching_results": teaching_results,
            "practice_results": practice_results,
            "final_result": final_result,
            "success_score": self._calculate_success_score(final_result)
        }
        
    def _run_teaching_phase(self) -> List[Dict]:
        results = []
        for _ in range(self.config.num_teaching_shots):
            result = self.transform_engine.generate_teaching_example()
            self.transform_engine.update_from_teaching(
                result,
                self.config.learning_rate
            )
            results.append(result)
        return results
        
    def _run_practice_phase(self) -> List[Dict]:
        results = []
        for _ in range(self.config.num_practice_shots):
            result = self.transform_engine.generate_practice_example()
            self.transform_engine.update_from_practice(
                result,
                self.config.learning_rate
            )
            results.append(result)
        return results
        
    def _run_target_phase(self, target_query: str) -> Dict:
        prev_loss = float('inf')
        for i in range(self.config.max_iterations):
            result = self.transform_engine.transform_query(target_query)
            curr_loss = result["loss"]
            
            if abs(prev_loss - curr_loss) < self.config.convergence_threshold:
                break
                
            self.transform_engine.update_from_target(
                result,
                self.config.learning_rate
            )
            prev_loss = curr_loss
            
        return result
        
    def _calculate_success_score(self, result: Dict) -> float:
        return np.clip(1.0 - result["loss"], 0.0, 1.0)
