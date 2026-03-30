import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from sgan.data.trajectories import TrajectoryDataset, seq_collate
from sgan.data.trajectories import HighDDataset, seq_collate_highd


def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader


def highd_data_loader(args, mmap_path, split='train'):
    """
    DataLoader for HighD mmap data produced by preprocess.py.

    Splitting strategy
    ------------------
    1. If args.highd_split_dir is set, load pre-computed index files from
       that directory (train_indices.npy / val_indices.npy / test_indices.npy)
       produced by split.py.

    2. Otherwise, split by recording ID using highd_val_ratio.

    Args
    ----
    args      : namespace with obs_len, pred_len, batch_size,
                loader_num_workers, use_I, highd_split_dir, highd_val_ratio
    mmap_path : path to the mmap directory
    split     : 'train' | 'val' | 'test'
    """
    mmap_path = Path(mmap_path)
    use_I  = getattr(args, 'use_I',  False)
    use_Iy = getattr(args, 'use_Iy', False)
    actual_path = mmap_path

    # ── determine sample indices for this split ───────────────────────────
    split_dir = getattr(args, 'highd_split_dir', '')
    if split_dir:
        # Load pre-computed indices from split.py
        idx_file = Path(split_dir) / f'{split}_indices.npy'
        if not idx_file.exists():
            raise FileNotFoundError(
                f"Split index file not found: {idx_file}\n"
                "Run data/highD/split.py first, or check --highd_split_dir."
            )
        indices = np.load(str(idx_file))
    else:
        # Split the single mmap by recording ID
        val_ratio    = getattr(args, 'highd_val_ratio', 0.1)
        rec_ids_path = actual_path / 'meta_recordingId.npy'
        if rec_ids_path.exists():
            rec_ids = np.load(str(rec_ids_path))
            unique_recs = np.unique(rec_ids)
            n_val_recs  = max(1, round(val_ratio * len(unique_recs)))
            val_recs    = set(unique_recs[-n_val_recs:])
            train_recs  = set(unique_recs) - val_recs
            if split == 'val':
                indices = np.where(np.isin(rec_ids, list(val_recs)))[0]
            else:
                indices = np.where(np.isin(rec_ids, list(train_recs)))[0]
        else:
            # Fallback: simple ratio split on sample indices
            N = np.load(str(actual_path / 'x_ego.npy'), mmap_mode='r').shape[0]
            n_val = max(1, round(val_ratio * N))
            if split == 'val':
                indices = np.arange(N - n_val, N)
            else:
                indices = np.arange(0, N - n_val)

    dset = HighDDataset(
        actual_path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        use_I=use_I,
        use_Iy=use_Iy,
        indices=indices,
    )

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=(split == 'train'),
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate_highd,
    )
    return dset, loader
