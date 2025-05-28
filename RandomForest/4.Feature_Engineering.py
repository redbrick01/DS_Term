import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

def data_preprocessing(is_plot=True):
  
    df = pd.read_csv("DS_term/RandomForest/3_Preprocessing_DataSet.csv", low_memory=False)
    df_test = pd.read_csv("DS_term/RandomForest/3_Preprocessing_TestDataSet.csv")

    df['load_ratio'] = df['Sload'] / (df['Dload'] + 1e-6)
    df['ttl_diff'] = abs(df['sttl'] - df['ct_state_ttl'])

    df_test['load_ratio'] = df_test['Sload'] / (df_test['Dload'] + 1e-6)
    df_test['ttl_diff'] = abs(df_test['sttl'] - df_test['ct_state_ttl'])

    X = df.drop("Label", axis=1)
    target = df["Label"]
    et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et_model.fit(X, target)

    et_importance = pd.Series(et_model.feature_importances_, index=X.columns)
    print(et_importance.sort_values(ascending=False).head(10))

    selected = et_importance[et_importance > 0.01].index.tolist()
    print(sorted(selected, reverse=True))
    df_final = X[selected]

    df_final.to_csv('DS_term/RandomForest/4_Feature_engineering_DataSet.csv', index=False)
    df_test.to_csv('DS_term/RandomForest/4_Feature_engineering_TestDataSet.csv', index=False)