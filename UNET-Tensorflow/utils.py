import pandas as pd
from dataset import PASCAL2007Dataset
from torch.utils.data import DataLoader

def get_colormap():
    d = {}
    df = pd.read_csv('colormap.csv', index_col=0)
    for label, r,g,b in zip(df.index, df['R'], df['G'], df['B']):
        d[label] = [int(r), int(g), int(b)]
    return d

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = PASCAL2007Dataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = PASCAL2007Dataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader