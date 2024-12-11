import json
import os
import glob
from collections import defaultdict

def asr(directory):
    """Analyze results from fixed size sweep experiments."""
    results = defaultdict(dict)
    
    result_files = glob.glob(os.path.join(directory, "*.json"))
    
    for file_path in result_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        header = data['HEADER']
        fixed_size = header['fixed_size']
        
        total_attempts = len(data['data'])
        successful = sum(1 for query_results in data['data'].values() 
                        if any('success' in trial and trial['success'] 
                              for trial in query_results))
        
        asr = successful / total_attempts if total_attempts > 0 else 0
        
        results[fixed_size] = {
            'asr': asr,
            'successful': successful,
            'total': total_attempts
        }
    
    return results

def main():
    sweep_dir = "insert your path here"
    results = asr(sweep_dir)
    
    print("\nFixed Size Sweep Results:")
    print("-" * 50)
    print(f"{'Size':<10} {'ASR':<10} {'Success/Total':<15}")
    print("-" * 50)
    
    for size in sorted(results.keys()):
        r = results[size]
        print(f"{size:<10} {r['asr']:.2%} {f'{r['successful']}/{r['total']}':<15}")

if __name__ == "__main__":
    main()
