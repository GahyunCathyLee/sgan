import argparse
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sgan.data.loader import data_loader, highd_data_loader
from sgan.data.trajectories import highd_nb_feat_dim
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path, bool_flag


# ─────────────────────────────────────
# AttrDict (Python 3.12 대응)
# ─────────────────────────────────────
class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value


# ─────────────────────────────────────
# Scenario label helpers
# ─────────────────────────────────────

def load_scenario_labels(path):
    path = Path(path)
    if not path.exists():
        print(f"[WARN] scenario_labels not found: {path}")
        return None
    df = pd.read_csv(path)
    required = {"recordingId", "trackId", "t0_frame"}
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] scenario_labels missing columns {missing}")
        return None
    has_event = "event_label" in df.columns
    has_state = "state_label" in df.columns
    if not has_event and not has_state:
        print("[WARN] scenario_labels has no event_label/state_label")
        return None
    lut = {}
    for row in df.itertuples(index=False):
        key = (int(row.recordingId), int(row.trackId), int(row.t0_frame))
        lut[key] = {
            "event_label": getattr(row, "event_label", None) if has_event else None,
            "state_label": getattr(row, "state_label", None) if has_state else None,
        }
    print(f"[INFO] Loaded scenario labels: {len(lut):,} entries from {path}")
    return lut


def build_sample_label_list(mmap_dir, indices, labels_lut):
    mmap_dir = Path(mmap_dir)
    meta_rec   = np.load(mmap_dir / "meta_recordingId.npy", mmap_mode='r')
    meta_track = np.load(mmap_dir / "meta_trackId.npy",     mmap_mode='r')
    meta_frame = np.load(mmap_dir / "meta_frame.npy",       mmap_mode='r')
    sample_labels = []
    for idx in indices:
        key = (int(meta_rec[idx]), int(meta_track[idx]), int(meta_frame[idx]))
        sample_labels.append(labels_lut.get(key))
    return sample_labels


def _sep(widths, left="+", mid="+", right="+", fill="-"):
    return left + mid.join(fill * w for w in widths) + right


def print_scenario_results(stats, label_type):
    if not stats:
        return
    rows = sorted(stats.items(), key=lambda x: (x[0] == "unknown", x[0]))
    c_lbl = max(len(lbl) for lbl, _ in rows)
    c_lbl = max(c_lbl, len(label_type)) + 2
    c_n = 9
    c_m = 11
    ws = [c_lbl, c_n, c_m, c_m, c_m]

    print(f"\n====== Scenario Results [{label_type}] ======")
    print(_sep(ws))
    print(f"|{label_type:^{c_lbl}}|{'n':^{c_n}}"
          f"|{'ADE':^{c_m}}|{'FDE':^{c_m}}|{'RMSE':^{c_m}}|")
    print(_sep(ws))

    for lbl, (sa, sf, sr, n) in rows:
        if n == 0:
            continue
        print(f"|{lbl:^{c_lbl}}|{n:^{c_n},}"
              f"|{sa/n:^{c_m}.4f}|{sf/n:^{c_m}.4f}|{sr/n:^{c_m}.4f}|")
    print(_sep(ws))

    total_sa = sum(v[0] for v in stats.values())
    total_sf = sum(v[1] for v in stats.values())
    total_sr = sum(v[2] for v in stats.values())
    total_n = sum(v[3] for v in stats.values())
    N = max(1, total_n)
    print(f"|{'Total':^{c_lbl}}|{total_n:^{c_n},}"
          f"|{total_sa/N:^{c_m}.4f}|{total_sf/N:^{c_m}.4f}|{total_sr/N:^{c_m}.4f}|")
    print(_sep(ws))


# ─────────────────────────────────────
# Argument
# ─────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str,
                    choices=['train', 'val', 'test'])
parser.add_argument('--gpu_num', default='0', type=str)

# HighD
parser.add_argument('--use_highd', default=1, type=bool_flag,
                    help='Override use_highd from checkpoint. '
                         'Set to 1 when evaluating a HighD-trained model.')
parser.add_argument('--highd_mmap_path', default='data/highD/mmap', type=str,
                    help='Override mmap path for evaluation.')
parser.add_argument('--highd_split_dir', default='data/highD/splits', type=str,
                    help='Override split directory for HighD evaluation.')

# Scenario
parser.add_argument('--scenario_labels', type=str, default=None,
                    help='Path to scenario_labels.csv for per-scenario breakdown')


# ─────────────────────────────────────
# RMSE
# ─────────────────────────────────────
@torch.no_grad()
def rmse(pred_abs: torch.Tensor, y_abs: torch.Tensor) -> torch.Tensor:
    """(B, T, 2) → (B,)"""
    return torch.norm(pred_abs - y_abs, dim=-1).pow(2).mean(dim=-1).sqrt()


# ─────────────────────────────────────
# Generator
# ─────────────────────────────────────
def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])

    nb_feat_dim = args.get('nb_feat_dim')
    if nb_feat_dim is None:
        nb_feat_dim = highd_nb_feat_dim(
            args.get('feature_mode', None),
            use_I=args.get('use_I', False),
            use_Iy=args.get('use_Iy', False),
            use_dim=args.get('use_dim', False),
        )

    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm,
        nb_feat_dim=nb_feat_dim,
        nb_K=8
    )

    if checkpoint.get('g_best_state') is not None:
        generator.load_state_dict(checkpoint['g_best_state'])
    else:
        generator.load_state_dict(checkpoint['g_state'])

    generator.cuda()
    generator.eval()
    return generator


# ─────────────────────────────────────
# Batch unpack
# ─────────────────────────────────────
def _unpack_eval_batch(args, batch):
    batch = [tensor.cuda(non_blocking=True) for tensor in batch]
    use_highd = getattr(args, 'use_highd', False)

    if use_highd:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
         non_linear_ped, loss_mask, seq_start_end,
         nb_feats, nb_mask) = batch
    else:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
         non_linear_ped, loss_mask, seq_start_end) = batch
        nb_feats = None
        nb_mask = None

    return (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
            non_linear_ped, loss_mask, seq_start_end, nb_feats, nb_mask)


# ─────────────────────────────────────
# Evaluation
# ─────────────────────────────────────
def evaluate(args, loader, generator, num_samples, sample_labels=None):

    total_ade = 0.0
    total_fde = 0.0
    total_rmse = 0.0
    total_traj = 0

    ev_stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    st_stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    sample_cursor = 0

    with torch.no_grad():
        for batch in loader:
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end,
             nb_feats, nb_mask) = _unpack_eval_batch(args, batch)

            B = pred_traj_gt.size(1)
            total_traj += B

            ade_list = []
            fde_list = []
            pred_list = []

            for _ in range(num_samples):
                pred_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end,
                    nb_feats, nb_mask
                )
                pred_abs = relative_to_abs(pred_rel, obs_traj[-1])

                ade_list.append(displacement_error(
                    pred_abs, pred_traj_gt, mode='raw'
                ))
                fde_list.append(final_displacement_error(
                    pred_abs[-1], pred_traj_gt[-1], mode='raw'
                ))
                pred_list.append(pred_abs)

            # best-of-K (ADE 기준)
            ade_stack = torch.stack(ade_list, dim=1)  # (B, K)
            best_idx = torch.argmin(ade_stack, dim=1)  # (B,)

            pred_stack = torch.stack(pred_list, dim=2)  # (T, B, K, 2)
            best_pred = pred_stack[:, torch.arange(B), best_idx]  # (T, B, 2)

            # ADE / FDE per sample
            best_ade = ade_stack[torch.arange(B), best_idx]  # (B,)
            fde_stack = torch.stack(fde_list, dim=1)
            best_fde = fde_stack[torch.arange(B), best_idx]  # (B,)

            total_ade += best_ade.sum()
            total_fde += best_fde.sum()

            # RMSE per sample
            best_pred_b = best_pred.permute(1, 0, 2)  # (B, T, 2)
            gt_b = pred_traj_gt.permute(1, 0, 2)
            rmse_batch = rmse(best_pred_b, gt_b)  # (B,)
            total_rmse += rmse_batch.sum()

            # Per-scenario accumulation
            if sample_labels is not None:
                ade_np = (best_ade / args.pred_len).cpu().numpy()
                fde_np = best_fde.cpu().numpy()
                rmse_np = rmse_batch.cpu().numpy()
                for i in range(B):
                    if sample_cursor + i >= len(sample_labels):
                        break
                    lab = sample_labels[sample_cursor + i]
                    if lab is None:
                        continue
                    ev = lab.get("event_label") or "unknown"
                    st = lab.get("state_label") or "unknown"
                    for acc, lbl in ((ev_stats, ev), (st_stats, st)):
                        acc[lbl][0] += float(ade_np[i])
                        acc[lbl][1] += float(fde_np[i])
                        acc[lbl][2] += float(rmse_np[i])
                        acc[lbl][3] += 1

            sample_cursor += B

    ade = total_ade / (total_traj * args.pred_len)
    fde = total_fde / total_traj
    rmse_val = total_rmse / total_traj

    return ade.item(), fde.item(), rmse_val.item(), total_traj, ev_stats, st_stats


# ─────────────────────────────────────
# Loader
# ─────────────────────────────────────
def build_loader(eval_args, cli_args):
    use_highd = eval_args.get('use_highd', False)

    if use_highd:
        mmap_path = eval_args.get('highd_mmap_path', 'data/highD/mmap')
        _, loader = highd_data_loader(
            eval_args, mmap_path, split=cli_args.dset_type
        )
        dataset_label = f'HighD-{cli_args.dset_type}'
    else:
        dset_path = get_dset_path(eval_args.dataset_name,
                                 cli_args.dset_type)
        _, loader = data_loader(eval_args, dset_path)
        dataset_label = f'{eval_args.dataset_name}-{cli_args.dset_type}'

    return loader, dataset_label


# ─────────────────────────────────────
# Main
# ─────────────────────────────────────
def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

    if os.path.isdir(args.model_path):
        paths = [
            os.path.join(args.model_path, f)
            for f in sorted(os.listdir(args.model_path))
            if f.endswith('.pt')
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        print(f'\n[Evaluating] {path}')

        checkpoint = torch.load(path, weights_only=False)
        generator = get_generator(checkpoint)

        eval_args = AttrDict(checkpoint['args'])

        if args.use_highd is not None:
            eval_args['use_highd'] = args.use_highd
        if args.highd_mmap_path is not None:
            eval_args['highd_mmap_path'] = args.highd_mmap_path
        if args.highd_split_dir is not None:
            eval_args['highd_split_dir'] = args.highd_split_dir

        loader, dataset_label = build_loader(eval_args, args)

        # Scenario labels
        sample_labels = None
        scenario_labels_path = args.scenario_labels
        if scenario_labels_path is None and eval_args.get('use_highd', False):
            mmap_path = eval_args.get('highd_mmap_path', 'data/highD/mmap')
            candidate = Path(mmap_path) / 'scenario_labels.csv'
            if candidate.exists():
                scenario_labels_path = str(candidate)

        if scenario_labels_path:
            labels_lut = load_scenario_labels(scenario_labels_path)
            if labels_lut is not None and eval_args.get('use_highd', False):
                mmap_path = eval_args.get('highd_mmap_path', 'data/highD/mmap')
                dset = loader.dataset
                sample_labels = build_sample_label_list(
                    mmap_path, dset.indices, labels_lut)

        ade, fde, rmse_val, total_traj, ev_stats, st_stats = evaluate(
            eval_args, loader, generator, args.num_samples, sample_labels
        )

        print(
            f'Dataset: {dataset_label} | #Traj: {total_traj} | '
            f'ADE: {ade:.4f} | FDE: {fde:.4f} | RMSE: {rmse_val:.4f}'
        )

        if ev_stats:
            print_scenario_results(ev_stats, label_type="Event")
        if st_stats:
            print_scenario_results(st_stats, label_type="State")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
