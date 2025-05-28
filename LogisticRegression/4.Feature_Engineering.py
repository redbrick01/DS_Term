import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2

def data_preprocessing(is_plot=True):
    df = pd.read_csv("DS_term/LogisticRegression/3_Preprocessing_DataSet.csv", low_memory=False)
    df_test = pd.read_csv("DS_term/LogisticRegression/3_Preprocessing_TestDataSet.csv")

    df['load_ratio'] = df['Sload'] / (df['Dload'] + 1e-6)
    df['ttl_diff'] = abs(df['sttl'] - df['ct_state_ttl'])

    df_test['load_ratio'] = df_test['Sload'] / (df_test['Dload'] + 1e-6)
    df_test['ttl_diff'] = abs(df_test['sttl'] - df_test['ct_state_ttl'])


    exclude_prefixes = ('proto_', 'state_', 'service_')
    excluded_cols = [col for col in df.columns if col.startswith(exclude_prefixes)]
    target_col = 'Label'

    df_onehot = df[excluded_cols] 
    df_numeric = df.drop(columns=excluded_cols, axis=1)
    df_target = df[target_col]


    print(df_onehot.dtypes.value_counts())
    print(df_numeric.dtypes.value_counts())

    selector = SelectKBest(score_func=chi2, k=10) 
    onehot_selected = selector.fit_transform(df_onehot, df_target)
    selected_cat_cols = df_onehot.columns[selector.get_support()]
    df_onehot_selected = df_onehot[selected_cat_cols]
    print("Number of 1st selected features(onehot)):", len(df_onehot_selected.columns))


    corr_target_cols = [col for col in df_numeric.columns]
    correlations = df_numeric.corr()[target_col].abs()
    selected_numeric = correlations[correlations > 0.3].index.tolist()
    df_correlations = df[selected_numeric]
    print("Number of 2nd selected features(numeric)):", len(df_correlations.columns))

    if is_plot:
        corr_matrix = df_correlations.corr().round(2)
        plt.figure(figsize=(30, 20))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, annot_kws={"size": 8})
        plt.title("Correlation Matrix (Filtered Features Only)")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("correlation.png", dpi=300, bbox_inches='tight')
        plt.show()

    df_final = pd.concat([df_onehot_selected, df_correlations], axis=1)
    print("Number of final selected features:", len(df_final.columns))
    print(df_final.columns)

    df_final.to_csv("DS_term/LogisticRegression/4_Feature_engineering_DataSet.csv", index=False)
    df_test.to_csv('DS_term/LogisticRegression/4_Feature_engineering_TestDataSet.csv', index=False)