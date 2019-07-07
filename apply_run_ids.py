"""
This script handles linking series together by using their orientation data. 
This is possible because through EDA we saw that the data was collected in 'runs'.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and process Data:
train_X = pd.read_csv(
    'data/X_train.csv').iloc[:, 3:].values.reshape(-1, 128, 10)
test_X = pd.read_csv('data/X_test.csv').iloc[:, 3:].values.reshape(-1, 128, 10)

full_X = pd.concat([pd.read_csv('data/X_train.csv').iloc[:, 3:], pd.read_csv(
    'data/X_test.csv').iloc[:, 3:]], axis=0).values.reshape(-1, 128, 10)

df_train_y = pd.read_csv('data/y_train.csv')
df_test_X = pd.read_csv('data/X_test.csv')

df_test_y = pd.DataFrame(
    np.arange(0, df_test_X.shape[0]/128), columns=['series_id'])
df_test_y['series_id'] = df_test_y['series_id'].apply(
    lambda x: x + 3810).astype(int)
df_test_y.set_index('series_id', append=False, inplace=True)
del df_test_y.index.name
df_test_y['series_id'] = df_test_y.index.astype(int)
df_test_y['group_id'] = df_test_y['series_id'].apply(lambda x: 99)
df_test_y['surface'] = df_test_y['series_id'].apply(lambda x: 'test')

# Combine test and train dataset (Note: may have been better to not do this).
df_full_y = pd.concat([df_train_y, df_test_y], axis=0)

# build a dict to convert surface names into numbers
surface_names = df_train_y['surface'].unique()
num_surfaces = len(surface_names)
surface_to_numeric = dict(zip(surface_names, range(num_surfaces)))

# y and group data as numeric values:
full_y = df_full_y['surface'].replace(
    surface_to_numeric).replace({'test': 10}).values
full_group = df_full_y['group_id'].values


def check(i, closest_i, data2, dist_list, n):
    """
    Function to check if series joined with itself (forward).

    :param i: Data to engineer features from (DataFrame).
    :param closest_i: Data to engineer features from (DataFrame).
    :param data2: Data to engineer features from (DataFrame).
    :param dist_list: Data to engineer features from (DataFrame).
    :param n: Data to engineer features from (DataFrame).
    """
    n += 1
    surface = data2.loc[i, 'surface']
    surface_closest = data2.loc[closest_i, 'surface']
    if closest_i == i:  # this might happen and it's definitely wrong
        print('Sample', i, 'linked with itself. Next closest sample used instead.')
        closest_i = np.argsort(dist_list)[n]
        closest_i = check(i, closest_i, data2, dist_list, n)
    return closest_i


def check_rev(closest_rev, i, closest_i, data2, dist_list, n):
    """
    Function to check if series joined with itself (backward).

    :param closest_rev: Data to engineer features from (DataFrame).
    :param i: Data to engineer features from (DataFrame).
    :param closest_i: Data to engineer features from (DataFrame).
    :param data2: Data to engineer features from (DataFrame).
    :param dist_list: Data to engineer features from (DataFrame).
    :param n: Data to engineer features from (DataFrame).
    """
    n += 1
    surface = data2.loc[i, 'surface']
    surface_closest = data2.loc[closest_i, 'surface']
    surface_rev = data2.loc[closest_rev, 'surface']
    if closest_i == closest_rev:  # this might happen and it's definitely wrong
        print('Sample', i, '(back-)linked with itself. Next closest sample used instead.')
        closest_rev = np.argsort(dist_list)[n]
        closest_rev = check_rev(closest_rev, i, closest_i, data2, dist_list, n)
    return closest_rev


def sq_dist(a, b):
    """
    Function to calculate the squared Euclidian distance between two series.

    :param a: Data to engineer features from (DataFrame).
    :param b: Data to engineer features from (DataFrame).
    """

    return np.sum((a-b)**2, axis=1)


def find_run_edges(data, data2, edge):
    """
    Function to find the edge of the run, ie the start/end of a run.

    :param data: Data to engineer features from (DataFrame).
    :param data2: Data to engineer features from (DataFrame).
    :param edge: Data to engineer features from (DataFrame).
    """
    if edge == 'left':
        border1 = 0
        border2 = -1
    elif edge == 'right':
        border1 = -1
        border2 = 0
    else:
        return False

    edge_list = []
    linked_list = []

    for i in range(len(data)):
        # distances to rest of samples
        dist_list = sq_dist(data[i, border1, :4], data[:, border2, :4])
        min_dist = np.min(dist_list)
        closest_i = np.argmin(dist_list)  # this is i's closest neighbor
        closest_i = check(i, closest_i, data2, dist_list, n=0)
        # now find closest_i's closest neighbor
        dist_list = sq_dist(data[closest_i, border2, :4], data[:, border1, :4])
        rev_dist = np.min(dist_list)
        closest_rev = np.argmin(dist_list)  # here it is
        closest_rev = check_rev(
            closest_rev, i, closest_i, data2, dist_list, n=0)
        if (i != closest_rev):  # we found an edge
            edge_list.append(i)
        else:
            linked_list.append([i, closest_i, min_dist])

    return edge_list, linked_list


def find_runs(data, left_edges, right_edges):
    """
    Function to link series to their closest neighbors to form a 'chain'.

    :param data: Data to engineer features from (DataFrame).
    :param left_edges: Data to engineer features from (DataFrame).
    :param right_edges: Data to engineer features from (DataFrame).
    """
    data_runs = []

    for start_point in left_edges:
        i = start_point
        run_list = [i]
        while i not in right_edges:
            tmp = np.argmin(sq_dist(data[i, -1, :4], data[:, 0, :4]))
            if tmp == i:  # self-linked sample
                tmp = np.argsort(sq_dist(data[i, -1, :4], data[:, 0, :4]))[1]
            i = tmp
            run_list.append(i)
        data_runs.append(np.array(run_list))

    return data_runs


full_left_edges, full_left_linked = find_run_edges(
    full_X, df_full_y, edge='left')
full_right_edges, full_right_linked = find_run_edges(
    full_X, df_full_y, edge='right')
print('Found', len(full_left_edges), 'left edges and',
      len(full_right_edges), 'right edges.')

full_runs = find_runs(full_X, full_left_edges, full_right_edges)

lost_samples = np.array([i for i in range(len(full_X))
                         if i not in np.concatenate(full_runs)])

find_run_edges(full_X[lost_samples], df_full_y.loc[lost_samples].reset_index(
    drop=True), edge='left')[1][0]

lost_run = np.array(lost_samples[find_runs(full_X[lost_samples], [0], [5])[0]])
full_runs.append(lost_run)

full_runs.append(full_runs[16][160:])
full_runs.append(full_runs[16][:160])
full_runs.append(full_runs[19][57:])
full_runs.append(full_runs[19][:57])

full_runs.append(full_runs[71][2:])
full_runs.append(full_runs[71][:2])
full_runs.append(full_runs[24][66:])
full_runs.append(full_runs[24][:66])

full_runs.append(full_runs[28][37:])
full_runs.append(full_runs[28][:37])
full_runs.append(full_runs[36][152:])
full_runs.append(full_runs[36][:152])
full_runs.append(full_runs[31][11:])
full_runs.append(full_runs[31][:11])

df_full_y['run_id'] = 0
df_full_y['run_pos'] = 0

for run_id in range(len(full_runs)):
    for run_pos in range(len(full_runs[run_id])):
        series_id = full_runs[run_id][run_pos]
        df_full_y.at[series_id, 'run_id'] = run_id
        df_full_y.at[series_id, 'run_pos'] = run_pos

df_full_y.to_csv('y_full_with_runs.csv', index=False)
df_full_y.tail()
