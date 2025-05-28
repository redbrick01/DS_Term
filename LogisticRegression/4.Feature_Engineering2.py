import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

def data_preprocessing(is_plot=True):
    df = pd.read_csv("DS_term/LogisticRegression/3_Preprocessing_DataSet.csv", low_memory=False)
    df_test = pd.read_csv("DS_term/LogisticRegression/3_Preprocessing_TestDataSet.csv")

    df['load_ratio'] = df['Sload'] / (df['Dload'] + 1e-6)
    df['ttl_diff'] = abs(df['sttl'] - df['ct_state_ttl'])

    df_test['load_ratio'] = df_test['Sload'] / (df_test['Dload'] + 1e-6)
    df_test['ttl_diff'] = abs(df_test['sttl'] - df_test['ct_state_ttl'])



    target_col = 'Label'

    X = df.drop(columns=[target_col])
    y = df[target_col]

    logit_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=10000, random_state=42)
    selector = SelectFromModel(logit_l1)
    selector.fit(X, y)

    selected_features = np.array(X.columns)[selector.get_support()]
    print(f"Number of selected features: {len(selected_features)}")
    print("Selected feature list:", selected_features)

    df.to_csv("DS_term/LogisticRegression/4_Feature_engineering_DataSet.csv", index=False)
    df_test.to_csv('DS_term/LogisticRegression/4_Feature_engineering_TestDataSet.csv', index=False)