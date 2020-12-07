import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import data


def main():
    root = data['root']

    # Videos validation split
    df = pd.read_csv(root / 'train_labels.csv')

    impact_count = df.groupby('gameKey')['impact'].sum()
    df_videos = pd.DataFrame(impact_count).reset_index()

    q25 = np.quantile(impact_count, q=0.25)
    q75 = np.quantile(impact_count, q=0.75)
    df_videos['quantile'] = 1
    df_videos.loc[df_videos['impact'] < q25, 'quantile'] = 2
    df_videos.loc[df_videos['impact'] > q75, 'quantile'] = 3

    df_videos['train'] = 1
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idxs, val_idxs in kfold.split(df_videos, df_videos['quantile']):
        df_videos.loc[val_idxs, 'train'] = 0
        break

    df_videos = df_videos.drop(['impact', 'quantile'], axis=1)
    df_videos.to_csv('validation_split.csv', index=False)

    # Images validation split
    df = pd.read_csv(root / 'image_labels.csv')
    df['gameKey'] = df['image'].apply(lambda x: x.split('_')[0])
    df['frame'] = df['image'].apply(lambda x: x.split('_')[-1][5:-4])

    df = df[['gameKey', 'frame']]
    df = df.drop_duplicates()
    counts = df.groupby('gameKey')['frame'].count()

    game_keys = counts.index
    game_counts = counts.values

    game_counts[game_counts < 12] = 0
    game_counts[(game_counts >= 12) & (game_counts < 17)] = 1
    game_counts[game_counts >= 17] = 2

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idxs, val_idxs in kfold.split(game_keys, game_counts):
        break

    new_df = pd.DataFrame(game_keys)
    new_df['train'] = 1
    new_df.loc[val_idxs, 'train'] = 0

    new_df.to_csv(root / 'images_validation_split.csv', index=False)
