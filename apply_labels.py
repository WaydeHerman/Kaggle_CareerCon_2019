"""
This script handles making a 'weak' prediction for each series using a RandomForest model.

"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Load Data:
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
sub = pd.read_csv("data/sample_submission.csv")


def feature_engineer(data):
    """
    Function to engineer new features.

    :param data: Data to engineer features from (DataFrame).
    """

    df = pd.DataFrame()
    data['totl_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 +
                              data['angular_velocity_Z']**2) ** 0.5
    data['totl_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 +
                             data['linear_acceleration_Z'])**0.5

    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']

    for col in data.columns:
        if col in ['row_id', 'series_id', 'measurement_number']:
            continue
        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()
        df[col + '_median'] = data.groupby(['series_id'])[col].median()
        df[col + '_max'] = data.groupby(['series_id'])[col].max()
        df[col + '_min'] = data.groupby(['series_id'])[col].min()
        df[col + '_std'] = data.groupby(['series_id'])[col].std()
        df[col + '_range'] = df[col + '_max'] - df[col + '_min']
        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']
        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(
            lambda x: np.mean(np.abs(np.diff(x))))
        df[col + '_abs_max'] = data.groupby(['series_id']
                                            )[col].apply(lambda x: np.max(np.abs(x)))
        df[col + '_abs_min'] = data.groupby(['series_id']
                                            )[col].apply(lambda x: np.min(np.abs(x)))
        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2
    return df


X_train = feature_engineer(X_train)
X_test = feature_engineer(X_test)

# Drop the following:
X_train = X_train.iloc[:, 44:].drop(columns=['acc_vs_vel_mean_abs_chg'])
X_test = X_test.iloc[:, 44:].drop(columns=['acc_vs_vel_mean_abs_chg'])

# Encode target:
le = LabelEncoder()
y_train['surface'] = le.fit_transform(y_train['surface'])

# Fill nans and infinite values:
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
X_train.replace(-np.inf, 0, inplace=True)
X_train.replace(np.inf, 0, inplace=True)
X_test.replace(-np.inf, 0, inplace=True)
X_test.replace(np.inf, 0, inplace=True)


def k_folds(X, y, X_test, k):
    """
    Function to make out of fold predictions.

    :param X: Train features (DataFrame).
    :param y: Train targets (Series).
    :param X_test: Test features (DataFrame).
    :param k: Number of folds (int).
    """

    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2019)
    y_test = np.zeros((X_test.shape[0], 9))
    y_oof = np.zeros((X.shape[0]))
    score = 0
    for i, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
        clf.fit(X_train.iloc[train_idx], y[train_idx])
        y_oof[val_idx] = clf.predict(X.iloc[val_idx])
        y_test += clf.predict_proba(X_test) / folds.n_splits
        score += clf.score(X.iloc[val_idx], y[val_idx])
        print('Fold: {} score: {}'.format(
            i, clf.score(X.iloc[val_idx], y[val_idx])))
    print('Avg Accuracy', score / folds.n_splits)

    return y_oof, y_test


y_oof, y_test = k_folds(X_train, y_train['surface'], X_test, k=50)

y_test = np.argmax(y_test, axis=1)
submission = sub
submission['surface'] = le.inverse_transform(y_test)
submission.to_csv('labels.csv', index=False)
submission.head(10)
