# Report
### Try RandomForestClassifier
randomForest_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=8,
        random_state=42,
        class_weight="balanced"
    )
ACC: 0.89709

### Try LGBMClassifier
lightGBM_clf = LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    metric='binary_logloss',
    learning_rate=0.01,
    max_depth=4,
)
ACC: 0.93959