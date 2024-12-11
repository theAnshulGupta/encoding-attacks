import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16

os.makedirs('plots', exist_ok=True)

with open("results/output.json", "r") as f:
    results_data = json.load(f)

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

# Prepare the data
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

# 1. Create main figure with subplots
fig = plt.figure(figsize=(20, 15))
gs = GridSpec(2, 2, figure=fig)
fig.suptitle('Bijection Learning Attack Analysis', fontsize=20, y=0.95)

# Plot 1: Success Rate by Intent with CI (Top Left)
ax1 = fig.add_subplot(gs[0, 0])
success_by_intent = trial_df.groupby('intent')['success'].agg(['mean', 'count'])
success_by_intent['ci'] = 1.96 * np.sqrt((success_by_intent['mean'] * (1 - success_by_intent['mean'])) / success_by_intent['count'])

# Sort by success rate
success_by_intent = success_by_intent.sort_values('mean', ascending=True)

# Create bar plot with error bars
colors = sns.color_palette("husl", len(success_by_intent))
bars = ax1.barh(success_by_intent.index, success_by_intent['mean'] * 100, color=colors)
ax1.errorbar(success_by_intent['mean'] * 100, success_by_intent.index,
             xerr=success_by_intent['ci'] * 100, fmt='none', color='black', capsize=5)

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
             f'{width:.1f}%\nn={int(success_by_intent["count"][i])}',
             va='center')

ax1.set_title('Attack Success Rate by Intent')
ax1.set_xlabel('Success Rate (%)')
ax1.set_xlim(0, 105)  # Leave space for labels

# Plot 2: Response Length Distribution (Top Right)
ax2 = fig.add_subplot(gs[0, 1])
sns.violinplot(data=trial_df, x='intent', y='response_length', 
               hue='success', ax=ax2, split=True, palette="Set2")
ax2.set_title('Response Length Distribution by Intent and Success')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.legend(title='Attack Success')
ax2.set_ylabel('Response Length (words)')

# Plot 3: Failure Mode Analysis (Bottom Left)
ax3 = fig.add_subplot(gs[1, 0])
failure_mode_counts = trial_df[trial_df['success'] == 0].groupby(['intent', 'failure_mode']).size().unstack(fill_value=0)
failure_mode_props = failure_mode_counts.div(failure_mode_counts.sum(axis=1), axis=0) * 100

# Create stacked bar plot with custom colors
failure_mode_props.plot(kind='barh', stacked=True, ax=ax3, 
                       colormap='Set3')
ax3.set_title('Failure Mode Distribution by Intent')
ax3.set_xlabel('Percentage of Failed Attempts')
ax3.legend(title='Failure Mode', bbox_to_anchor=(1.05, 1))

# Plot 4: Query Length Analysis (Bottom Right)
ax4 = fig.add_subplot(gs[1, 1])
sns.boxplot(data=trial_df, x='intent', y='query_length', ax=ax4, palette="Set2")
ax4.set_title('Query Length Distribution by Intent')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
ax4.set_ylabel('Query Length (words)')

# Adjust layout
plt.tight_layout()
plt.savefig('plots/publication_figure.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a separate success rate trend visualization
plt.figure(figsize=(12, 6))
trial_numbers = range(1, 6)  # 5 trials per query
success_by_trial = pd.DataFrame()

for intent in trial_df['intent'].unique():
    intent_data = trial_df[trial_df['intent'] == intent]
    trial_success = [intent_data[intent_data['trial_index'] == i]['success'].mean() * 100 for i in trial_numbers]
    success_by_trial[intent] = trial_success

success_by_trial.index = trial_numbers
ax = success_by_trial.plot(marker='o', linewidth=2, markersize=8)
plt.title('Attack Success Rate Trend Across Trials')
plt.xlabel('Trial Number')
plt.ylabel('Success Rate (%)')
plt.grid(True, alpha=0.3)
plt.legend(title='Intent', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('plots/success_rate_trend.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated publication-ready plots:")
print("1. plots/publication_figure.png - Main figure with four subplots")
print("2. plots/success_rate_trend.png - Success rate trends across trials")

# Print summary statistics
print("\nSummary Statistics:")
print("\nSuccess Rate by Intent:")
success_stats = trial_df.groupby('intent')['success'].agg(['mean', 'std', 'count'])
success_stats['mean'] = success_stats['mean'] * 100
success_stats['std'] = success_stats['std'] * 100
print(success_stats.round(2).sort_values('mean', ascending=False))

print("\nAverage Response Length by Success:")
response_stats = trial_df.groupby(['intent', 'success'])['response_length'].mean().round(1)
print(response_stats)
