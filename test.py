## code cell

import numpy as np
#feature train data
ftrain_df=pd.read_csv('./train_data/features_train.tsv',sep='\t')
#drop colums loss more than half data
nan_keys=ftrain_df.isna().sum()[ftrain_df.isna().sum().values>=len(ftrain_df)*0.5].keys()
ftrain_df=ftrain_df.drop(nan_keys,axis=1)
ftrain_df.head()
# %%