import argparse
import pandas as pd
from scipy.stats import ttest_rel


def main():
    parser = argparse.ArgumentParser(description='Statistical significance test')
    parser.add_argument('--result_csv', type=str, required=True, help='Path to results CSV')
    parser.add_argument('--target_method', type=str, required=True, help='Target method for comparison')
    parser.add_argument('--compare_methods', type=str, nargs='+', required=True, help='Methods to compare')
    parser.add_argument('--metric', type=str, default='accuracy', help='Metric to compare')
    parser.add_argument('--save_path', type=str, required=True, help='Save path for results')
    args = parser.parse_args()
    
    df = pd.read_csv(args.result_csv)
    
    target_scores = df[df['method'] == args.target_method][args.metric].values
    
    results = []
    
    for method in args.compare_methods:
        compare_scores = df[df['method'] == method][args.metric].values
        
        if len(target_scores) != len(compare_scores):
            print(f"Warning: Different number of samples for {method}")
            continue
        
        t_stat, p_value = ttest_rel(target_scores, compare_scores)
        
        results.append({
            'comparison': f"{args.target_method} vs {method}",
            't_stat': t_stat,
            'p_value': p_value
        })
        
        print(f"{args.target_method} vs {method}\t p = {p_value:.3f}")
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(args.save_path, index=False)


if __name__ == '__main__':
    main()