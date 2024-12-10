import json
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load results
results_file = "results/small_test_sonnet_new.json"
with open(results_file, "r") as f:
    results_data = json.load(f)

# Extract intents from queries
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

# Analyze results by intent
intent_stats = defaultdict(lambda: {'total': 0, 'successful': 0, 'trials': []})

for query, trials in results_data["data"].items():
    intent = get_intent(query)
    for trial in trials:
        if 'human_judge' not in trial:
            continue
        intent_stats[intent]['total'] += 1
        if trial['human_judge'] == 'unsafe':
            intent_stats[intent]['successful'] += 1
        intent_stats[intent]['trials'].append(trial)

# Convert to DataFrame for plotting
plot_data = []
for intent, stats in intent_stats.items():
    if stats['total'] > 0:
        asr = (stats['successful'] / stats['total']) * 100
        plot_data.append({
            'Intent': intent,
            'ASR': asr,
            'Total Trials': stats['total'],
            'Successful': stats['successful']
        })

df = pd.DataFrame(plot_data)

# Create visualization
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Main ASR plot
ax = sns.barplot(data=df, x='Intent', y='ASR')
plt.title('Attack Success Rate by Intent Category')
plt.ylabel('Attack Success Rate (%)')
plt.xlabel('')
plt.xticks(rotation=45)

# Add trial counts on top of bars
for i, row in df.iterrows():
    ax.text(i, row['ASR'], 
            f'n={row["Total Trials"]}', 
            ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Print detailed statistics
print("\nDetailed Statistics by Intent:")
print(df.sort_values('ASR', ascending=False).to_string(index=False))

print("\nOverall Statistics:")
print(f"Total Trials: {df['Total Trials'].sum()}")
print(f"Average ASR: {df['ASR'].mean():.2f}%")
print(f"Weighted ASR: {(df['Successful'].sum() / df['Total Trials'].sum() * 100):.2f}%")
