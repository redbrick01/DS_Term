import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

df = pd.read_csv("DS_term/LinearRegression/3_Preprocessing_DataSet.csv", low_memory=False)

df['load_ratio'] = df['Sload'] / (df['Dload'] + 1e-6)
df['ttl_diff'] = abs(df['sttl'] - df['ct_state_ttl'])

target_col = 'Label'

X = df.drop(columns=[target_col])
y = df[target_col]

lasso = Lasso(alpha=0.001, max_iter=10000, random_state=42)
lasso.fit(X, y)
selected_features = np.array(X.columns)[lasso.coef_ != 0]

print(f"Number of selected features: {len(selected_features)}")
print("Selected feature list:", selected_features)


df.to_csv("DS_term/LinearRegression/4_Feature_engineering_DataSet.csv", index=False)