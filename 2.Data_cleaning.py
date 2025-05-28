import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("DS_term/1_Raw_DataSet.csv", low_memory=False)
original_count = df.shape[0]


df['ct_flw_http_mthd'] = df['ct_flw_http_mthd'].fillna(0).astype(np.uint8)
df['is_ftp_login'] = df['is_ftp_login'].fillna(0).astype(np.uint8)
df['ct_ftp_cmd'] = pd.to_numeric(df['ct_ftp_cmd'], errors='coerce').fillna(0).astype('Int64')


df_no_dup = df.drop_duplicates().copy()
after_dup_count = df_no_dup.shape[0]
duplicates_removed = original_count - after_dup_count


for col in df_no_dup.columns:
    unique_vals = df_no_dup[col].dropna().unique()
    if set(unique_vals).issubset({0, 1}):
        df_no_dup[col] = df_no_dup[col].astype(np.uint8)


outlier_cols = ['tcprtt', 'synack', 'ackdat']
mask = pd.Series(True, index=df_no_dup.index)
for col in outlier_cols:
    q1 = df_no_dup[col].quantile(0.25)
    q3 = df_no_dup[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask &= df_no_dup[col].between(lower, upper)

df_cleaned = df_no_dup[mask]
after_outlier_count = df_cleaned.shape[0]
outliers_removed = after_dup_count - after_outlier_count


columns_to_drop = ['srcip', 'dstip', 'Stime', 'Ltime', 'stcpb', 'dtcpb']
df_cleaned = df_cleaned.drop(columns=columns_to_drop, errors='ignore')


df_cleaned.to_csv("DS_term/2_Cleaning_DataSet.csv", index=False)


sizes = [duplicates_removed, outliers_removed, after_outlier_count]
labels = ['Duplicates removed', 'Outliers removed', 'Final remain']
colors = ['lightcoral', 'lightskyblue', 'lightgray']

plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Data Cleaning Breakdown')
plt.axis('equal')
plt.show()
