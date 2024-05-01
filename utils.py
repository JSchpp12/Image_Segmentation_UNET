import pandas as pd
from dataset import PASCAL2007Dataset
from torch.utils.data import DataLoader
import numpy as np

def get_colormap():
    d = {}
    df = pd.read_csv('colormap.csv', index_col=0)
    for label, r,g,b in zip(df.index, df['R'], df['G'], df['B']):
        d[label] = [int(r), int(g), int(b)]
    return d


def convert_label_to_rgb(label_2d):
    d = get_colormap()
    labels = list(d.keys())
    img = np.zeros((label_2d.shape[0], label_2d.shape[1], 3))
    for i_r, r in enumerate(label_2d):
        for i_c, c in enumerate(r):
                img[i_r][i_c] = d[labels[c]]

    return img
