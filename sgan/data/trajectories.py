import logging
import os
import math
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# HighD dataset
# ─────────────────────────────────────────────────────────────────────────────

# x_nb feature indices (neighformer preprocess.py schema, 10 dims):
#   dx(0) dy(1) dvx(2) dvy(3) dax(4) day(5) s_x(6) s_y(7) dim(8) I(9)
_NB_BASE_INDICES = [0, 1, 2, 3, 4, 5]
_NB_DIM_INDEX = 8    # vehicle size bin (0~4)
_NB_I_INDEX   = 9    # composite importance
_NB_FEATURE_MODES = {
    'baseline': _NB_BASE_INDICES,
    'dimI': _NB_BASE_INDICES + [_NB_DIM_INDEX, _NB_I_INDEX],
}


def highd_nb_feat_indices(feature_mode=None, use_I=False, use_Iy=False, use_dim=False):
    if feature_mode is not None:
        if feature_mode not in _NB_FEATURE_MODES:
            choices = ', '.join(sorted(_NB_FEATURE_MODES))
            raise ValueError(f"Unknown feature_mode={feature_mode!r}. Choose one of: {choices}")
        return list(_NB_FEATURE_MODES[feature_mode])

    # Legacy fallback for old checkpoints/commands.
    extra = []
    if use_dim:
        extra.append(_NB_DIM_INDEX)
    if use_I or use_Iy:
        extra.append(_NB_I_INDEX)
    return _NB_BASE_INDICES + extra


def highd_nb_feat_dim(feature_mode=None, use_I=False, use_Iy=False, use_dim=False):
    return len(highd_nb_feat_indices(feature_mode, use_I=use_I, use_Iy=use_Iy, use_dim=use_dim))


class HighDDataset(Dataset):
    """
    Dataset that reads memory-mapped arrays produced by preprocess.py.

    Each sample corresponds to one ego vehicle scenario:
      obs : ego (x, y) history of length obs_len
      pred: ego (x, y) future  of length pred_len
      nb_feats : neighbor features (obs_len, K, nb_feat_dim)
                 baseline = [dx, dy, dvx, dvy, dax, day]
                 dimI = baseline + [dim, I]
      nb_mask  : (K,) bool — True if that neighbor slot is ever occupied

    The batch collated by seq_collate_highd has the same first-7-field
    layout as the original seq_collate so that discriminator_step /
    generator_step can share most of their code:

      obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
      non_linear_ped, loss_mask, seq_start_end,
      nb_feats, nb_mask
    """

    def __init__(self, mmap_path, obs_len=None, pred_len=None,
                 feature_mode=None, use_I=False, use_Iy=False, use_dim=False,
                 indices=None, threshold=0.002):
        super().__init__()
        mmap_path = Path(mmap_path)

        # Memory-mapped arrays (read-only, zero copy)
        self.x_ego   = np.load(mmap_path / 'x_ego.npy',   mmap_mode='r')  # (N, T, 6)
        self.y       = np.load(mmap_path / 'y.npy',        mmap_mode='r')  # (N, Tf, 2)
        self.x_nb    = np.load(mmap_path / 'x_nb.npy',    mmap_mode='r')  # (N, T, K, 10)
        self.nb_mask = np.load(mmap_path / 'nb_mask.npy', mmap_mode='r')  # (N, T, K)

        T_mmap  = self.x_ego.shape[1]
        Tf_mmap = self.y.shape[1]

        self.obs_len  = obs_len  if obs_len  is not None else T_mmap
        self.pred_len = pred_len if pred_len is not None else Tf_mmap

        if self.obs_len != T_mmap or self.pred_len != Tf_mmap:
            raise ValueError(
                f"Requested obs_len={self.obs_len}, pred_len={self.pred_len} "
                f"but mmap has T={T_mmap}, Tf={Tf_mmap}. "
                "Re-run preprocess.py with matching --history_sec / --future_sec, "
                "or omit --obs_len / --pred_len to use mmap dimensions."
            )

        self.feature_mode = feature_mode
        self.use_I     = use_I
        self.use_Iy    = use_Iy
        self.use_dim   = use_dim
        self.threshold = threshold
        self.nb_feat_indices = highd_nb_feat_indices(
            feature_mode, use_I=use_I, use_Iy=use_Iy, use_dim=use_dim)

        N = self.x_ego.shape[0]
        self.indices = np.asarray(indices) if indices is not None else np.arange(N)

    @property
    def nb_feat_dim(self):
        return len(self.nb_feat_indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])

        # ── ego trajectory (x, y) ─────────────────────────────────────────
        x_ego = np.array(self.x_ego[idx], dtype=np.float32)   # (T, 6)
        obs_xy = x_ego[:, :2]                                  # (T, 2)

        obs_rel = np.zeros_like(obs_xy)
        obs_rel[1:] = obs_xy[1:] - obs_xy[:-1]

        # ── future trajectory ──────────────────────────────────────────────
        y = np.array(self.y[idx], dtype=np.float32)            # (Tf, 2)

        pred_rel = np.zeros_like(y)
        pred_rel[0]  = y[0] - obs_xy[-1]
        pred_rel[1:] = y[1:] - y[:-1]

        # ── non-linear flag (same poly_fit as TrajectoryDataset) ───────────
        # poly_fit expects shape (2, seq_len)
        nl = poly_fit(obs_xy.T, self.obs_len, self.threshold)

        # ── neighbor features ──────────────────────────────────────────────
        x_nb_raw = np.array(self.x_nb[idx], dtype=np.float32)       # (T, K, 10)
        nb_feats = x_nb_raw[:, :, self.nb_feat_indices]              # (T, K, F)

        mask_raw = np.array(self.nb_mask[idx])                       # (T, K) bool
        nb_mask  = mask_raw.any(axis=0)                              # (K,)

        # ── loss mask (all ones — no padding for HighD samples) ───────────
        loss_mask = np.ones(self.obs_len + self.pred_len, dtype=np.float32)

        return (
            torch.from_numpy(obs_xy).float(),      # (T, 2)
            torch.from_numpy(y).float(),            # (Tf, 2)
            torch.from_numpy(obs_rel).float(),      # (T, 2)
            torch.from_numpy(pred_rel).float(),     # (Tf, 2)
            torch.tensor(nl).float(),               # scalar
            torch.from_numpy(nb_feats).float(),     # (T, K, F)
            torch.from_numpy(nb_mask),              # (K,) bool
            torch.from_numpy(loss_mask).float(),    # (T+Tf,)
        )


def seq_collate_highd(data):
    """
    Collate for HighDDataset.

    Returns a 9-tuple that extends the original seq_collate 7-tuple:
      obs_traj       (T_obs,  B, 2)
      pred_traj      (T_pred, B, 2)
      obs_traj_rel   (T_obs,  B, 2)
      pred_traj_rel  (T_pred, B, 2)
      non_linear_ped (B,)
      loss_mask      (B, T_obs+T_pred)
      seq_start_end  (B, 2)   — each scenario has exactly 1 ego → trivial
      nb_feats       (T_obs, B, K, F)
      nb_mask        (B, K)
    """
    (obs_list, pred_list, obs_rel_list, pred_rel_list,
     nl_list, nb_feats_list, nb_mask_list, loss_mask_list) = zip(*data)

    B = len(obs_list)
    seq_start_end = torch.LongTensor([[i, i + 1] for i in range(B)])

    obs_traj      = torch.stack(obs_list,      dim=0).permute(1, 0, 2)   # (T, B, 2)
    pred_traj     = torch.stack(pred_list,     dim=0).permute(1, 0, 2)   # (Tf, B, 2)
    obs_traj_rel  = torch.stack(obs_rel_list,  dim=0).permute(1, 0, 2)
    pred_traj_rel = torch.stack(pred_rel_list, dim=0).permute(1, 0, 2)
    non_linear    = torch.stack(nl_list,       dim=0)                    # (B,)
    loss_mask     = torch.stack(loss_mask_list, dim=0)                   # (B, T+Tf)
    nb_feats      = torch.stack(nb_feats_list,  dim=0).permute(1, 0, 2, 3)  # (T, B, K, F)
    nb_mask       = torch.stack(nb_mask_list,   dim=0)                   # (B, K)

    return (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
            non_linear, loss_mask, seq_start_end,
            nb_feats, nb_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Original ETH/UCY dataset
# ─────────────────────────────────────────────────────────────────────────────

def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end
    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :]
        ]
        return out
