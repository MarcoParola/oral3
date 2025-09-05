import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Configuration ---
INPUT_FILE_PATH = r"E:\TESI\OFFICIAL_ORAL3\oral3\logs\metrics_logs\ExperimentMetrics_aug_perc.csv"
OUTPUT_PLOT_PATH = r"E:\TESI\OFFICIAL_ORAL3\oral3\logs\metrics_logs\metrics_plot_aug_percentage.png"
Z_SCORE = 1.645  # 90% confidence interval

METRICS_MAP = {
    'Accuracy': 'Test/Accuracy',
    'F1-Score': 'Test/F1-score',
    'Precision': 'Test/Precision',
    'Recall': 'Test/Recall'
}

# --- Main Script ---
df = pd.read_csv(INPUT_FILE_PATH)

COL_PERCENTAGE = 'Aug. Percentage'
percentages = sorted(df[COL_PERCENTAGE].unique())

# Restructure data for plotting
all_metrics_data = {}
for metric_name, column_name in METRICS_MAP.items():
    pivot_df = df.pivot_table(index='Seed', columns=COL_PERCENTAGE, values=column_name)
    all_metrics_data[metric_name] = pivot_df.values

# Determine dynamic y-axis limits
global_min, global_max = np.inf, -np.inf
for metric_name in METRICS_MAP.keys():
    data = all_metrics_data[metric_name]
    
    # Calculate stats using standardized variable names
    mean_value = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0, ddof=1)
    num_samples = data.shape[0]
    margin_of_error = Z_SCORE * (std_dev / np.sqrt(num_samples))
    
    current_min = np.min(mean_value - margin_of_error)
    current_max = np.max(mean_value + margin_of_error)
    if current_min < global_min: global_min = current_min
    if current_max > global_max: global_max = current_max

padding = (global_max - global_min) * 0.05
y_min_limit = global_min - padding
y_max_limit = global_max + padding

# --- Plot Generation ---
metric_colors = {
    'Accuracy': '#1f77b4',
    'F1-Score': '#d62728',
    'Precision': '#ff7f0e',
    'Recall': '#2ca02c'
}
plt.style.use('seaborn-v0_8-whitegrid')

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
plt.ylim(y_min_limit, y_max_limit)

for ax, metric_name in zip(axes.flatten(), METRICS_MAP.keys()):
    data = all_metrics_data[metric_name]
    color = metric_colors[metric_name]

    mean_value = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0, ddof=1)
    num_samples = data.shape[0]
    margin_of_error = Z_SCORE * (std_dev / np.sqrt(num_samples))

    ax.plot(percentages, mean_value, color=color, marker='o',
            linestyle='-', linewidth=2, markersize=6, label=f'Mean {metric_name}')

    ax.fill_between(percentages,
                    mean_value - margin_of_error,
                    mean_value + margin_of_error,
                    color=color, alpha=0.2, label='90% Confidence Interval')

    ax.set_title(metric_name, fontsize=14, weight='bold')
    ax.legend(loc='lower right')

# Finalize and save the plot
fig.supxlabel('Percentage of Synthetic Data Used (%)', fontsize=16)
fig.supylabel('Performance Metric Value', fontsize=16)
fig.suptitle('Trend of Performance Metrics vs. Data Augmentation', fontsize=20, weight='bold')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
plt.show()