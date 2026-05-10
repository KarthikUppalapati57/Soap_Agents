import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to the JSON file
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "results", "Final_rouge_summary.json")

# Load the data
with open(json_path, 'r') as f:
    data = json.load(f)

# Extract data for plotting
labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
zero_shot = [data['zero_shot']['rouge1'], data['zero_shot']['rouge2'], data['zero_shot']['rougeL']]
one_shot = [data['one_shot']['rouge1'], data['one_shot']['rouge2'], data['one_shot']['rougeL']]
few_shot = [data['few_shot']['rouge1'], data['few_shot']['rouge2'], data['few_shot']['rougeL']]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, zero_shot, width, label='Zero-Shot', color='#1f77b4')
rects2 = ax.bar(x, one_shot, width, label='One-Shot', color='#ff7f0e')
rects3 = ax.bar(x + width, few_shot, width, label='Few-Shot', color='#2ca02c')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('ROUGE Scores by Prompting Strategy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3, fmt='%.4f')
ax.bar_label(rects2, padding=3, fmt='%.4f')
ax.bar_label(rects3, padding=3, fmt='%.4f')

fig.tight_layout()

# Save the plot
output_path = os.path.join(base_dir, "results", "rouge_scores_comparison.png")
plt.savefig(output_path, dpi=300)
print(f"Graph successfully saved to {output_path}")
