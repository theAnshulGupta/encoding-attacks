import argparse
from typing import Dict, List
from execution.processor import ExecutionConfig, Executor
from execution.optimizer import OptimizationConfig, ParameterOptimizer
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Languages")
    
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--teaching-shots", type=int, default=5)
    parser.add_argument("--practice-shots", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--transform-type", type=str, default="word",
                       choices=["word", "contextual"])
                       
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--opt-teaching", type=int, nargs="+",
                       default=[3, 5, 7])
    parser.add_argument("--opt-practice", type=int, nargs="+",
                       default=[1, 2, 3])
    parser.add_argument("--opt-lr", type=float, nargs="+",
                       default=[0.05, 0.1, 0.2])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-dataset", action="store_true", default=True)
                       
    return parser.parse_args()

def load_target_queries(use_dataset: bool = True) -> List[str]:
    """Load target queries either from dataset or use default examples"""
    if use_dataset:
        # Load queries from HarmBench dataset
        df = pd.read_csv('harm.csv')
        return df['Behavior'].tolist()[:10]  # Start with first 10 queries for testing
    else:
        # Default example queries
        return [
            "What is the weather like today?",
            "Tell me about machine learning",
            "How does a computer work?"
        ]

def main():
    args = parse_args()
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    base_config = ExecutionConfig(
        model_name=args.model,
        num_teaching_shots=args.teaching_shots,
        num_practice_shots=args.practice_shots,
        learning_rate=args.learning_rate
    )
    
    target_queries = load_target_queries(args.use_dataset)
    print(f"Loaded {len(target_queries)} queries from {'HarmBench dataset' if args.use_dataset else 'default examples'}")
    
    if args.optimize:
        opt_config = OptimizationConfig(
            num_teaching_shots=args.opt_teaching,
            num_practice_shots=args.opt_practice,
            learning_rates=args.opt_lr,
            transformation_types=["word", "contextual"],
            num_workers=args.num_workers
        )
        
        optimizer = ParameterOptimizer(opt_config, base_config)
        results = optimizer.run_optimization(target_queries)
        
        print("\nParameter Optimization Results:")
        print(f"Overall Success Rate: {results['analysis']['overall_success_rate']:.2%}")
        print("\nBest Parameters:")
        for param, value in results['analysis']['best_params'].items():
            print(f"{param}: {value}")
            
    else:
        executor = Executor(base_config)
        executor.initialize_transformation(args.transform_type)
        
        results = []
        for i, query in enumerate(target_queries, 1):
            print(f"\nProcessing query {i}/{len(target_queries)}")
            result = executor.run_execution(query)
            results.append(result)
            print(f"Success Score: {result['success_score']:.2%}")
            
        print("\nExecution Results Summary:")
        avg_score = sum(r['success_score'] for r in results) / len(results)
        print(f"Average Success Score: {avg_score:.2%}")

if __name__ == "__main__":
    main()
