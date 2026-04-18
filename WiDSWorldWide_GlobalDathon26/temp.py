import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
import os
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
warnings.filterwarnings('ignore')

#Load data
train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
meta = pd.read_csv('Data/metaData.csv')

#Convert to pd.Dataframe
train = pd.DataFrame(train)
test = pd.DataFrame(test)
train.head()

X_train = train.drop(columns=['event', 'event_id', 'time_to_hit_hours'])
y_train = train[['time_to_hit_hours', 'event']]
y_train['event'] = y_train['event'].astype(int)
y_train_surv = Surv.from_arrays(event=y_train["event"].astype(bool), 
                           time=y_train["time_to_hit_hours"])
event_val = y_train['event'].copy()
#Testing data
X_test = test.drop(columns=['event_id'])
event_id = test['event_id']

category_col = [
    'event_start_hour',
    'event_start_dayofweek',
    'event_start_month'
]
gbsa = GradientBoostingSurvivalAnalysis(
    n_estimators=100, 
    learning_rate=0.05, 
    max_depth=3, 
    random_state=42
)
#Initialize oof loop
N_FOLDS = 5
skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=42
)
#out-of-fold prediction
oof_preds = np.zeros((len(X_train), 4)) # 4 horizons: 12h, 24h, 48h, 72h
#final average prediction for X_test
test_preds = np.zeros((len(X_test), 4))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, event_val)):
    #Splitting data
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr_surv = y_train_surv[train_idx]
    y_tr = event_val[train_idx]
    X_test_fold = X_test.copy()
    #Target encoding for category data (train fold)
    target_encoder = TargetEncoder(cols=category_col, smoothing=10)
    X_tr[category_col] = target_encoder.fit_transform(X_tr[category_col], y_tr)
    X_val[category_col] = target_encoder.transform(X_val[category_col])
    X_test_fold = target_encoder.transform(X_test_fold[category_col])
    #Train each fold
    gbsa.fit(X_tr, y_tr)
    surv_fn = gbsa.predict_survival_function(X_val)
    horizons = [12, 24, 28 , 72]
    #filling oof matrix
    for i, sf in enumerate(surv_fn):
        for j, horizon in enumerate(horizons):
            stamp = min(horizon, sf.x[-1])
            oof_preds[] = 1 - sf(stamp)

