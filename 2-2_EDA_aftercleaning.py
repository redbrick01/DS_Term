import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("DS_term/2_Cleaning_DataSet.csv",  low_memory=False)

numeric_cols = df.select_dtypes(include='number').columns
output_dir = "DS_term/cleaned_eda_histograms"
os.makedirs(output_dir, exist_ok=True)

plt.style.use('ggplot')

for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    
    plt.hist(df[df['Label'] == 0][col].dropna(), bins=50,
             color='#4a90e2', edgecolor='white', alpha=0.7, label='Normal')
    plt.title(f'Distribution of {col} (Normal Only)', fontsize=16, fontweight='bold')
    plt.xlabel(col, fontsize=13)
    plt.ylabel('Count', fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"{col}_hist_normal_only.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

