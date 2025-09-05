import numpy as np
import pandas as pd
import os

# --- Configuration ---
INPUT_FILE_PATH = r"E:\TESI\OFFICIAL_ORAL3\oral3\logs\metrics_logs\ExperimentMetrics.csv"
OUTPUT_DIR_PATH = r"E:\TESI\OFFICIAL_ORAL3\oral3\logs\metrics_logs\confidence_intervals"
Z_SCORE = 1.645  # 90% confidence interval

METRICS_MAP = {
    'Accuracy': 'Test/Accuracy',
    'F1-Score': 'Test/F1-score',
    'Precision': 'Test/Precision',
    'Recall': 'Test/Recall'
}

# Abbreviated names for output files
AUGMENTATION_MAP = {
    'No augmentation': 'No aug.',
    'Traditional augmentation': 'Tr. aug.',
    'SD + text': 'SD(TXT)',
    'SD + text + img': 'SD(TXT+IMG)',
    'StyleGAN3': 'SG3(LBL)',
    'AC-GAN': 'AC-SG3'
}

# --- Main Script ---
try:
    df = pd.read_csv(INPUT_FILE_PATH)

    group_keys = ['Dataset', 'Model', 'Augmentation']
    for (dataset, model, aug_type), group in df.groupby(group_keys):
        
        summary_data = []
        # Iterate through the map to use consistent column names
        for metric_name, column_name in METRICS_MAP.items():
            if column_name not in group:
                continue

            values = group[column_name].dropna()
            num_samples = len(values)
            if num_samples == 0:
                continue

            # Calculate statistics using standardized variable names
            mean_value = values.mean()
            std_dev = values.std(ddof=1) if num_samples > 1 else 0
            margin_of_error = Z_SCORE * (std_dev / np.sqrt(num_samples)) if num_samples > 0 else 0

            summary_data.append({
                "Metric": metric_name,
                "n": num_samples,
                "Mean": mean_value,
                "Error (±)": margin_of_error,
                "CI_low": mean_value - margin_of_error,
                "CI_high": mean_value + margin_of_error,
                "Mean ± Error": f"{mean_value:.6f} $\\pm$ {margin_of_error:.6f}"
            })

        if not summary_data:
            continue

        results_df = pd.DataFrame(summary_data)
        
        output_path = os.path.join(OUTPUT_DIR_PATH, dataset, model)
        os.makedirs(output_path, exist_ok=True)
        
        file_name = AUGMENTATION_MAP.get(aug_type, aug_type)
        full_path = os.path.join(output_path, f"{file_name}.csv")
        results_df.to_csv(full_path, index=False, float_format="%.6f")

except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILE_PATH}' not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")