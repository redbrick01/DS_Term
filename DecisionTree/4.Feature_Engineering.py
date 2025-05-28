import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

# Load the dataset
df = pd.read_csv("DS_term/DecisionTree/3_Preprocessing_DataSet.csv", low_memory=False)

# feature creation - load_ratio
df['load_ratio'] = df['Sload'] / (df['Dload'] + 1e-6)
# feature creation - ttl_diff
df['ttl_diff'] = abs(df['sttl'] - df['ct_state_ttl'])

X = df.drop("Label", axis=1)
target = df["Label"]
et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_model.fit(X, target)

et_importance = pd.Series(et_model.feature_importances_, index=X.columns)
print(et_importance.sort_values(ascending=False).head(10))

selected = et_importance[et_importance > 0.01].index.tolist()
print(sorted(selected, reverse=True))
df_final = X[selected]

df_final.to_csv('DS_term/DecisionTree/4_Feature_engineering_DataSet.csv', index=False)