import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("DS_term/2_Cleaning_DataSet.csv", low_memory=False)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col not in ['Label']]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


df.to_csv("DS_term/DecisionTree/3_Preprocessing_DataSet.csv", index=False)

