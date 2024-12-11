import json
from collections import defaultdict
import pandas as pd

results_file = "results/output.json"
with open(results_file, "r") as f:
    results_data = json.load(f)

# 1. Query-level statistics
query_stats = []
for query, trials in results_data["data"].items():
    successful = 0
    total = 0
    failure_modes = defaultdict(int)
    
    for trial in trials:
        if 'person' not in trial:
            continue
        total += 1
        if trial['person'] == 'unsafe':
            successful += 1
        
        if 'failure_mode_judge' in trial:
            failure_modes[trial['failure_mode_judge']] += 1
    
    if total > 0:
        query_stats.append({
            'query': query,
            'successful_trials': successful,
            'total_trials': total,
            'asr': (successful / total * 100),
            'failure_ok': failure_modes.get('ok', 0) / total * 100,
            'failure_refusal': failure_modes.get('refusal', 0) / total * 100,
            'failure_irrelevant': failure_modes.get('irrelevant', 0) / total * 100,
            'failure_other': failure_modes.get('other', 0) / total * 100
        })

query_df = pd.DataFrame(query_stats)
query_df.to_csv('results/query_level_stats.csv', index=False)

# 2. Intent-level statistics
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

intent_stats = defaultdict(lambda: {'total': 0, 'successful': 0, 'failure_modes': defaultdict(int)})

for query, trials in results_data["data"].items():
    intent = get_intent(query)
    for trial in trials:
        if 'person' not in trial:
            continue
        intent_stats[intent]['total'] += 1
        if trial['person'] == 'unsafe':
            intent_stats[intent]['successful'] += 1
        if 'failure_mode_judge' in trial:
            intent_stats[intent]['failure_modes'][trial['failure_mode_judge']] += 1

intent_data = []
for intent, stats in intent_stats.items():
    if stats['total'] > 0:
        total = stats['total']
        intent_data.append({
            'intent': intent,
            'successful_trials': stats['successful'],
            'total_trials': total,
            'asr': (stats['successful'] / total * 100),
            'failure_ok': stats['failure_modes'].get('ok', 0) / total * 100,
            'failure_refusal': stats['failure_modes'].get('refusal', 0) / total * 100,
            'failure_irrelevant': stats['failure_modes'].get('irrelevant', 0) / total * 100,
            'failure_other': stats['failure_modes'].get('other', 0) / total * 100
        })

intent_df = pd.DataFrame(intent_data)
intent_df.to_csv('results/intent_level_stats.csv', index=False)

# 3. Category-level statistics (from previous analysis)
def get_category(query):
    query = query.lower()
    if any(word in query for word in ['generate', 'write']):
        if any(word in query for word in ['lyrics', 'passage', 'story']):
            return 'Content Generation'
        elif any(word in query for word in ['code', 'function', 'script']):
            return 'Code Generation'
        else:
            return 'Text Generation'
    elif any(word in query for word in ['instructions', 'guide', 'steps']):
        return 'Instructions'
    return 'Other'

category_stats = defaultdict(lambda: {'total': 0, 'successful': 0, 'failure_modes': defaultdict(int)})

for query, trials in results_data["data"].items():
    category = get_category(query)
    for trial in trials:
        if 'person' not in trial:
            continue
        category_stats[category]['total'] += 1
        if trial['person'] == 'unsafe':
            category_stats[category]['successful'] += 1
        if 'failure_mode_judge' in trial:
            category_stats[category]['failure_modes'][trial['failure_mode_judge']] += 1

category_data = []
for category, stats in category_stats.items():
    if stats['total'] > 0:
        total = stats['total']
        category_data.append({
            'category': category,
            'successful_trials': stats['successful'],
            'total_trials': total,
            'asr': (stats['successful'] / total * 100),
            'failure_ok': stats['failure_modes'].get('ok', 0) / total * 100,
            'failure_refusal': stats['failure_modes'].get('refusal', 0) / total * 100,
            'failure_irrelevant': stats['failure_modes'].get('irrelevant', 0) / total * 100,
            'failure_other': stats['failure_modes'].get('other', 0) / total * 100
        })

category_df = pd.DataFrame(category_data)
category_df.to_csv('results/category_level_stats.csv', index=False)

# 4. Overall statistics
overall_stats = {
    'metric': [
        'Total Queries',
        'Total Trials',
        'Successful Trials',
        'Overall ASR',
        'Average Query ASR',
        'Average Intent ASR',
        'Average Category ASR'
    ],
    'value': [
        len(query_df),
        query_df['total_trials'].sum(),
        query_df['successful_trials'].sum(),
        (query_df['successful_trials'].sum() / query_df['total_trials'].sum() * 100),
        query_df['asr'].mean(),
        intent_df['asr'].mean(),
        category_df['asr'].mean()
    ]
}
overall_df = pd.DataFrame(overall_stats)
overall_df.to_csv('results/overall_stats.csv', index=False)

print("Results exported to CSV files in the results directory:")
print("1. query_level_stats.csv - Statistics for each individual query")
print("2. intent_level_stats.csv - Statistics aggregated by intent")
print("3. category_level_stats.csv - Statistics aggregated by category")
print("4. overall_stats.csv - Overall experiment statistics")
