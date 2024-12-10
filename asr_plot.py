import json
import glob
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]

# %%
# Load results
results_file = "results/small_test_sonnet_new.json"
with open(results_file, "r") as f:
    results_data = json.load(f)

# Calculate success rate and analyze failure modes for each query
query_stats = {}

for behavior, trials in results_data["data"].items():
    successful = 0
    total = 0
    failure_modes = {
        'ok': 0,
        'refusal': 0,
        'irrelevant': 0,
        'other': 0
    }
    
    for trial in trials:
        if 'human_judge' not in trial:
            continue
        total += 1
        if trial['human_judge'] == 'unsafe':
            successful += 1
        
        # Track failure modes
        if 'failure_mode_judge' in trial:
            mode = trial['failure_mode_judge']
            if mode in failure_modes:
                failure_modes[mode] += 1
            else:
                failure_modes['other'] += 1
    
    if total > 0:
        query_stats[behavior] = {
            'successful': successful,
            'total': total,
            'asr': (successful / total * 100),
            **{f'failure_{k}': v/total*100 for k,v in failure_modes.items()}
        }

# Convert to DataFrame
plot_data = []
for query, stats in query_stats.items():
    # Categorize queries
    category = 'Other'
    if any(word in query.lower() for word in ['generate', 'write']):
        if any(word in query.lower() for word in ['lyrics', 'passage', 'story']):
            category = 'Content Generation'
        elif any(word in query.lower() for word in ['code', 'function', 'script']):
            category = 'Code Generation'
        else:
            category = 'Text Generation'
    elif any(word in query.lower() for word in ['instructions', 'guide', 'steps']):
        category = 'Instructions'
    
    plot_data.append({
        'query': query[:50] + "..." if len(query) > 50 else query,
        'category': category,
        'asr': stats['asr'],
        'total_trials': stats['total'],
        **{k: v for k,v in stats.items() if k.startswith('failure_')}
    })
plot_df = pd.DataFrame(plot_data)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: ASR by Category
sns.boxplot(data=plot_df, x='category', y='asr', ax=ax1)
ax1.set_title('Attack Success Rate by Query Category')
ax1.set_ylabel('Attack Success Rate (%)')
ax1.set_xlabel('')
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Failure Mode Distribution
failure_cols = [col for col in plot_df.columns if col.startswith('failure_')]
failure_data = plot_df[failure_cols].mean()
failure_data = pd.DataFrame({'Mode': [col.replace('failure_', '').capitalize() for col in failure_cols], 
                           'Percentage': failure_data.values})

sns.barplot(data=failure_data, x='Mode', y='Percentage', ax=ax2)
ax2.set_title('Distribution of Failure Modes')
ax2.set_ylabel('Percentage of Trials (%)')
ax2.set_xlabel('')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Print statistics
print("\nSuccess Rate by Category:")
category_stats = plot_df.groupby('category').agg({
    'asr': ['mean', 'std', 'count'],
    'total_trials': 'sum'
}).round(2)
print(category_stats)

print("\nOverall Statistics:")
print(f"Overall ASR: {plot_df['asr'].mean():.2f}%")
print(f"Total Queries: {len(plot_df)}")
print(f"Total Trials: {plot_df['total_trials'].sum()}")

# Print failure mode analysis
print("\nFailure Mode Analysis:")
failure_mode_stats = plot_df[failure_cols].mean().round(2)
print(failure_mode_stats)
