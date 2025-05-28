import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

df = pd.read_csv("DS_term/2_Cleaning_DataSet.csv", low_memory=False)

log_transform_cols = [
   'dur','sbytes','dbytes','sttl','dttl','sloss','dloss',
   'Sload','Dload','Spkts','Dpkts','swin','dwin',
   'smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit',
   'Sintpkt','Dintpkt','ct_state_ttl',
   'ct_flw_http_mthd','ct_ftp_cmd','ct_srv_src','ct_srv_dst',
   'ct_dst_ltm','ct_src_ltm','ct_dst_src_ltm', 'tcprtt', 'synack', 'ackdat'
]

df[log_transform_cols] = df[log_transform_cols].apply(lambda x: np.log1p(x))

df_normal = df[df['Label'] == 0].copy()
df_attack = df[df['Label'] == 1].copy()

standard_scaler = StandardScaler()
standard_scaler.fit(df_normal[log_transform_cols])
df[log_transform_cols] = standard_scaler.transform(df[log_transform_cols])

df = df.drop(['sport', 'dsport'], axis=1)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col not in ['Label']]
df_final= pd.get_dummies(df, columns=categorical_cols)
bool_cols = df_final.select_dtypes(include=['bool']).columns
df_final[bool_cols] = df_final[bool_cols].astype(np.uint8)


print(df_final.dtypes.value_counts())
df_final.to_csv("DS_term/LinearRegression/3_Preprocessing_DataSet.csv", index=False)

