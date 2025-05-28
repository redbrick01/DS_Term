import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("DS_term/1_Raw_DataSet.csv",  low_memory=False)

numeric_cols = df.select_dtypes(include='number').columns
output_dir = "DS_term/raw_eda_histograms"
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

output_dir = "DS_term/raw_eda_boxplots"
os.makedirs(output_dir, exist_ok=True)

for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    
    sns.boxplot(x='Label', y=col, data=df, palette={'0': '#4a90e2', '1': '#e74c3c'}, 
                fliersize=2.5, linewidth=1.5, width=0.5)
    
    plt.title(f'Boxplot of {col} by Label', fontsize=16, fontweight='bold')
    plt.xlabel('Label (0 = Normal, 1 = Attack)', fontsize=13)
    plt.ylabel(col, fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"{col}_box_by_label.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
