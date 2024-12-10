import argparse
from typing import Dict, List
from execution.processor import ExecutionConfig, Executor
from execution.optimizer import OptimizationConfig, ParameterOptimizer

def parse_args():
    parser = argparse.ArgumentParser(description="Run language transformation executions")
    
    parser.add_argument("--model", type=str, required=True,
                       help="Target model name")
    parser.add_argument("--teaching-shots", type=int, default=5,
                       help="Number of teaching examples")
    parser.add_argument("--practice-shots", type=int, default=2,
                       help="Number of practice examples")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                       help="Learning rate for context adaptation")
    parser.add_argument("--transform-type", type=str, default="word",
                       choices=["word", "contextual"],
                       help="Type of transformation")
                       
    parser.add_argument("--optimize", action="store_true",
                       help="Run parameter optimization")
    parser.add_argument("--opt-teaching", type=int, nargs="+",
                       default=[3, 5, 7],
                       help="Teaching shots to optimize over")
    parser.add_argument("--opt-practice", type=int, nargs="+",
                       default=[1, 2, 3],
                       help="Practice shots to optimize over")
    parser.add_argument("--opt-lr", type=float, nargs="+",
                       default=[0.05, 0.1, 0.2],
                       help="Learning rates to optimize over")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of parallel workers for optimization")
                       
    return parser.parse_args()

def load_target_queries() -> List[str]:
    return [
        "What is the weather like today?",
        "Tell me about machine learning",
        "How does a computer work?"
    ]

def main():
    args = parse_args()
    
    base_config = ExecutionConfig(
        model_name=args.model,
        num_teaching_shots=args.teaching_shots,
        num_practice_shots=args.practice_shots,
        learning_rate=args.learning_rate
    )
    
    target_queries = load_target_queries()
    
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
        for query in target_queries:
            result = executor.run_execution(query)
            results.append(result)
            
        print("\nExecution Results:")
        for result in results:
            print(f"\nQuery: {result['query']}")
            print(f"Success Score: {result['success_score']:.2%}")

if __name__ == "__main__":
    main()
