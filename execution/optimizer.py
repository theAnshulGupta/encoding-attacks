# Standard library imports
import json
import logging
from typing import Dict, List, Optional, Tuple
import itertools
from concurrent.futures import ProcessPoolExecutor

# Third-party imports
import numpy as np
from tqdm import tqdm

# Local imports
from .processor import ExecutionConfig, Executor
from dataclasses import dataclass

class OptimizationConfig:
    num_teaching_shots: List[int]
    num_practice_shots: List[int]
    learning_rates: List[float]
    transformation_types: List[str]
    num_workers: int = 4

class ParameterOptimizer:
    def __init__(self, opt_config: OptimizationConfig, base_config: ExecutionConfig):
        self.opt_config = opt_config
        self.base_config = base_config
        
    def run_optimization(self, target_queries: List[str]) -> Dict[str, List[Dict]]:
        param_combinations = self._generate_param_combinations()
        results = self._run_parallel_executions(param_combinations, target_queries)
        analysis = self._analyze_results(results)
        return {
            "raw_results": results,
            "analysis": analysis
        }
        
    def _generate_param_combinations(self) -> List[Dict]:
        param_space = {
            "num_teaching_shots": self.opt_config.num_teaching_shots,
            "num_practice_shots": self.opt_config.num_practice_shots,
            "learning_rate": self.opt_config.learning_rates,
            "transformation_type": self.opt_config.transformation_types
        }
        combinations = []
        for values in itertools.product(*param_space.values()):
            param_dict = dict(zip(param_space.keys(), values))
            combinations.append(param_dict)
        return combinations
        
    def _run_parallel_executions(self, 
                               param_combinations: List[Dict],
                               target_queries: List[str]) -> List[Dict]:
        results = []
        with ProcessPoolExecutor(max_workers=self.opt_config.num_workers) as executor:
            futures = []
            for params in param_combinations:
                for query in target_queries:
                    future = executor.submit(
                        self._run_single_execution,
                        params,
                        query
                    )
                    futures.append(future)
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Execution failed: {str(e)}")
        return results
        
    def _run_single_execution(self, params: Dict, query: str) -> Dict:
        config = ExecutionConfig(
            model_name=self.base_config.model_name,
            num_teaching_shots=params["num_teaching_shots"],
            num_practice_shots=params["num_practice_shots"],
            learning_rate=params["learning_rate"]
        )
        executor = Executor(config)
        executor.initialize_transformation(params["transformation_type"])
        result = executor.run_execution(query)
        result.update(params)
        return result
        
    def _analyze_results(self, results: List[Dict]) -> Dict:
        analysis = {
            "overall_success_rate": self._calculate_success_rate(results),
            "success_by_param": self._analyze_by_parameter(results),
            "best_params": self._find_best_params(results)
        }
        return analysis
        
    def _calculate_success_rate(self, results: List[Dict]) -> float:
        successes = sum(1 for r in results if r["success_score"] > 0.5)
        return successes / len(results)
        
    def _analyze_by_parameter(self, results: List[Dict]) -> Dict:
        param_analysis = {}
        for param in ["num_teaching_shots", "num_practice_shots", 
                     "learning_rate", "transformation_type"]:
            param_analysis[param] = {}
            for result in results:
                value = result[param]
                if value not in param_analysis[param]:
                    param_analysis[param][value] = []
                param_analysis[param][value].append(result["success_score"])
            for value in param_analysis[param]:
                scores = param_analysis[param][value]
                param_analysis[param][value] = sum(scores) / len(scores)
        return param_analysis
        
    def _find_best_params(self, results: List[Dict]) -> Dict:
        param_results = {}
        for result in results:
            params = tuple(result[p] for p in ["num_teaching_shots", 
                                             "num_practice_shots",
                                             "learning_rate",
                                             "transformation_type"])
            if params not in param_results:
                param_results[params] = []
            param_results[params].append(result["success_score"])
        best_params = max(param_results.items(),
                         key=lambda x: sum(x[1])/len(x[1]))
        return {
            "num_teaching_shots": best_params[0][0],
            "num_practice_shots": best_params[0][1],
            "learning_rate": best_params[0][2],
            "transformation_type": best_params[0][3],
            "success_rate": sum(best_params[1])/len(best_params[1])
        }
