import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load results
results_file = "results/output.json"
with open(results_file, "r") as f:
    results_data = json.load(f)

# Helper functions
def get_intent(query):
    query = query.lower()
    if any(word in query for word in ['lyrics', 'passage', 'story']):
        return 'Content Creation'
    if any(word in query for word in ['code', 'script', 'function']):
        return 'Code Generation'
    if any(word in query for word in ['instructions', 'guide', 'steps', 'how to']):
        return 'Instructions'
    if any(word in query for word in ['threat', 'attack', 'hack', 'exploit']):
        return 'Harmful Actions'
    if any(word in query for word in ['discriminat', 'hate', 'insult']):
        return 'Hate Speech'
    return 'Other'

# 1. Prepare detailed trial-level data
trial_data = []
for query, trials in results_data["data"].items():
    intent = get_intent(query)
    query_length = len(query.split())
    
    for trial_idx, trial in enumerate(trials):
        if 'person' not in trial:
            continue
            
        success = 1 if trial['person'] == 'unsafe' else 0
        response_length = len(trial.get('response', '').split())
        
        trial_data.append({
            'query': query,
            'intent': intent,
            'trial_index': trial_idx,
            'success': success,
            'query_length': query_length,
            'response_length': response_length,
            'failure_mode': trial.get('failure_mode_judge', 'unknown')
        })

trial_df = pd.DataFrame(trial_data)

# 2. Create visualizations

# Set style
sns.set_style("whitegrid")
colors = sns.color_palette("husl", 8)

# Figure 1: Success Rate Distribution with Confidence Intervals
plt.figure(figsize=(12, 6))
success_by_intent = trial_df.groupby('intent')['success'].agg(['mean', 'count'])
success_by_intent['ci'] = 1.96 * np.sqrt((success_by_intent['mean'] * (1 - success_by_intent['mean'])) / success_by_intent['count'])

plt.errorbar(success_by_intent.index, success_by_intent['mean'] * 100,
             yerr=success_by_intent['ci'] * 100, fmt='o', capsize=5)
plt.title('Attack Success Rate by Intent (with 95% CI)')
plt.ylabel('Success Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/success_rate_with_ci.png')
plt.close()

# Figure 2: Query Length vs Success Rate
plt.figure(figsize=(10, 6))
sns.regplot(data=trial_df, x='query_length', y='success', 
            scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
plt.title('Query Length vs Attack Success')
plt.xlabel('Query Length (words)')
plt.ylabel('Success Rate')
plt.tight_layout()
plt.savefig('plots/query_length_vs_success.png')
plt.close()

# Figure 3: Response Length Distribution by Success
plt.figure(figsize=(10, 6))
sns.boxplot(data=trial_df, x='intent', y='response_length', hue='success')
plt.title('Response Length Distribution by Intent and Success')
plt.xticks(rotation=45)
plt.ylabel('Response Length (words)')
plt.legend(title='Attack Success')
plt.tight_layout()
plt.savefig('plots/response_length_distribution.png')
plt.close()

# Figure 4: Failure Mode Analysis
plt.figure(figsize=(12, 6))
failure_mode_counts = trial_df[trial_df['success'] == 0].groupby(['intent', 'failure_mode']).size().unstack(fill_value=0)
failure_mode_props = failure_mode_counts.div(failure_mode_counts.sum(axis=1), axis=0) * 100
failure_mode_props.plot(kind='bar', stacked=True)
plt.title('Failure Mode Distribution by Intent')
plt.ylabel('Percentage of Failed Attempts')
plt.xlabel('')
plt.legend(title='Failure Mode', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('plots/failure_mode_analysis.png')
plt.close()

# Statistical Analysis
print("\n=== Statistical Analysis ===")

# 1. Chi-square test for independence between intent and success
contingency = pd.crosstab(trial_df['intent'], trial_df['success'])
chi2, p_value = stats.chi2_contingency(contingency)[:2]
print(f"\nChi-square test (Intent vs Success):")
print(f"chi2 = {chi2:.2f}, p = {p_value:.4f}")

# 2. Query Length Analysis
successful_lengths = trial_df[trial_df['success'] == 1]['query_length']
failed_lengths = trial_df[trial_df['success'] == 0]['query_length']
t_stat, p_value = stats.ttest_ind(successful_lengths, failed_lengths)
print(f"\nQuery Length T-test (Successful vs Failed):")
print(f"t = {t_stat:.2f}, p = {p_value:.4f}")

# 3. Success Rate Statistics
print("\nSuccess Rate by Intent:")
success_stats = trial_df.groupby('intent')['success'].agg(['mean', 'std', 'count'])
success_stats['ci'] = 1.96 * np.sqrt((success_stats['mean'] * (1 - success_stats['mean'])) / success_stats['count'])
print(success_stats.round(3))

# Save detailed statistics to CSV
success_stats.to_csv('results/detailed_success_stats.csv')

# 4. Effect Size Analysis
print("\nEffect Size Analysis (Cohen's h between each intent and overall mean):")
overall_success_rate = trial_df['success'].mean()
for intent in trial_df['intent'].unique():
    intent_success_rate = trial_df[trial_df['intent'] == intent]['success'].mean()
    h = 2 * (np.arcsin(np.sqrt(intent_success_rate)) - np.arcsin(np.sqrt(overall_success_rate)))
    print(f"{intent}: {h:.3f}")

# Save trial-level data for future analysis
trial_df.to_csv('results/trial_level_analysis.csv', index=False)

print("\nAnalysis complete! Generated files:")
print("1. plots/success_rate_with_ci.png - Success rates with confidence intervals")
print("2. plots/query_length_vs_success.png - Query length vs success analysis")
print("3. plots/response_length_distribution.png - Response length distributions")
print("4. plots/failure_mode_analysis.png - Failure mode analysis")
print("5. results/detailed_success_stats.csv - Detailed statistics")
print("6. results/trial_level_analysis.csv - Trial-level data")
