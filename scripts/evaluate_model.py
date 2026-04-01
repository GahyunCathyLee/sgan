import argparse
import os
import torch

from sgan.data.loader import data_loader, highd_data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path, bool_flag


# ─────────────────────────────────────────
# AttrDict (Python 3.12 대응)
# ─────────────────────────────────────────
class AttrDict(dict):
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value


# ─────────────────────────────────────────
# Argument
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# RMSE (네가 원하는 방식)
# ─────────────────────────────────────────
@torch.no_grad()
def rmse(pred_abs: torch.Tensor, y_abs: torch.Tensor) -> torch.Tensor:
    """(B, T, 2) → (B,)"""
    return torch.norm(pred_abs - y_abs, dim=-1).pow(2).mean(dim=-1).sqrt()


# ─────────────────────────────────────────
# Generator
# ─────────────────────────────────────────
def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])

    nb_feat_dim = args.get('nb_feat_dim',
                           6
                           + (1 if args.get('use_dim', False) else 0)
                           + (1 if (args.get('use_I', False) or args.get('use_Iy', False)) else 0))

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


# ─────────────────────────────────────────
# Batch unpack
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# Evaluation (핵심)
# ─────────────────────────────────────────
def evaluate(args, loader, generator, num_samples):

    total_ade = 0.0
    total_fde = 0.0
    total_rmse = 0.0
    total_traj = 0

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

            # K 샘플 생성
            for _ in range(num_samples):
                pred_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end,
                    nb_feats, nb_mask
                )
                pred_abs = relative_to_abs(pred_rel, obs_traj[-1])

                ade_list.append(displacement_error(
                    pred_abs, pred_traj_gt, mode='raw'
                ))  # (B,)

                fde_list.append(final_displacement_error(
                    pred_abs[-1], pred_traj_gt[-1], mode='raw'
                ))  # (B,)

                pred_list.append(pred_abs)  # (T, B, 2)

            # ───────── best-of-K (ADE 기준)
            ade_stack = torch.stack(ade_list, dim=1)  # (B, K)
            best_idx = torch.argmin(ade_stack, dim=1)  # (B,)

            # ───────── best trajectory 선택
            pred_stack = torch.stack(pred_list, dim=2)  # (T, B, K, 2)

            best_pred = pred_stack[:, torch.arange(B), best_idx]  # (T, B, 2)

            # ───────── ADE / FDE
            total_ade += ade_stack[torch.arange(B), best_idx].sum()
            fde_stack = torch.stack(fde_list, dim=1)
            total_fde += fde_stack[torch.arange(B), best_idx].sum()

            # ───────── RMSE (네 방식)
            best_pred_b = best_pred.permute(1, 0, 2)  # (B, T, 2)
            gt_b = pred_traj_gt.permute(1, 0, 2)

            rmse_batch = rmse(best_pred_b, gt_b)  # (B,)
            total_rmse += rmse_batch.sum()

    ade = total_ade / (total_traj * args.pred_len)
    fde = total_fde / total_traj
    rmse_val = total_rmse / total_traj

    return ade.item(), fde.item(), rmse_val.item(), total_traj


# ─────────────────────────────────────────
# Loader
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
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

        ade, fde, rmse_val, total_traj = evaluate(
            eval_args, loader, generator, args.num_samples
        )

        print(
            f'Dataset: {dataset_label} | #Traj: {total_traj} | '
            f'ADE: {ade:.4f} | FDE: {fde:.4f} | RMSE: {rmse_val:.4f}'
        )


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)