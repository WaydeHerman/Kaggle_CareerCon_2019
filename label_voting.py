"""
This script handles linking series together by using their orientation data. 
This is possible because through EDA we saw that the data was collected in 'runs'.

"""

import pandas as pd
import numpy as np

submission_df = pd.read_csv('kernel_rf_submission.csv')
y_targets = pd.read_csv('data/y_train.csv')
X_train_df = pd.read_csv('data/X_train.csv')
full_y_df = pd.read_csv('y_full_with_runs.csv')

submission_df['series_id'] = submission_df['series_id'].apply(
    lambda x: x + 3810).astype(int)
submission_df.set_index('series_id', append=False, inplace=True)
del submission_df.index.name
submission_df['series_id'] = submission_df.index.astype(int)

test_y_df = full_y_df.iloc[int(X_train_df.shape[0]/128):, :]

for index, row in submission_df.iterrows():
    series_id = row['series_id']
    predictions = row['surface']
    full_y_df.loc[series_id, 'predictions'] = predictions

full_y_df.fillna('train', inplace=True)

# Voting using the labels from the classification and the 'runs
for i in range(0, full_y_df['run_id'].unique().shape[0]):
    if full_y_df[full_y_df['run_id'] == i]['surface'].unique().shape[0] >= 2:
        if 'test' in full_y_df[full_y_df['run_id'] == i]['surface'].unique():
            target = full_y_df[full_y_df['run_id'] == i]['surface'].unique()[0]
            full_y_df.loc[full_y_df.run_id == i, 'surface'] = target
    if full_y_df[full_y_df['run_id'] == i]['surface'].unique().shape[0] == 1:
        if 'test' in full_y_df[full_y_df['run_id'] == i]['surface'].unique():
            if full_y_df[full_y_df['run_id'] == i]['predictions'].unique().shape[0] == 1:
                full_y_df.loc[full_y_df.run_id == i,
                              'surface'] = full_y_df.loc[full_y_df.run_id == i, 'predictions']
            else:
                if full_y_df[full_y_df['run_id'] == i]['predictions'].value_counts()[0] != full_y_df[full_y_df['run_id'] == i]['predictions'].value_counts()[1]:
                    full_y_df.loc[full_y_df.run_id == i, 'surface'] = full_y_df[full_y_df['run_id']
                                                                                == i]['predictions'].value_counts().index[0]
                else:
                    full_y_df.loc[full_y_df.run_id == i,
                                  'surface'] = full_y_df.loc[full_y_df.run_id == i, 'predictions']

# Final submissions:
submission_df = full_y_df[['series_id', 'surface']
                          ].iloc[int(X_train_df.shape[0]/128):, :]

submission_df['series_id'] = submission_df['series_id'].apply(
    lambda x: x - 3810).astype(int)

submission_df.set_index('series_id', append=False, inplace=True)
del submission_df.index.name
submission_df['series_id'] = submission_df.index.astype(int)

submission_df = submission_df[['series_id', 'surface']]

submission_df.to_csv('final_submission.csv', index=False)
