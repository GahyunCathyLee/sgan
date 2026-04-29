import argparse
import gc
import logging
import os
import sys

from collections import defaultdict

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from sgan.data.loader import data_loader, highd_data_loader
from sgan.losses import gan_g_loss, gan_d_loss, l2_loss
from sgan.losses import displacement_error, final_displacement_error

from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path
from sgan.data.trajectories import highd_nb_feat_dim

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='zara1', type=str)
parser.add_argument('--delim', default=' ')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=200, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=1024, type=int)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=64, type=int)
parser.add_argument('--decoder_h_dim_g', default=128, type=int)
parser.add_argument('--noise_dim', default=None, type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='ped')
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--g_learning_rate', default=1e-4, type=float)
parser.add_argument('--g_steps', default=1, type=int)

# Pooling Options
parser.add_argument('--pooling_type', default='pool_net')
parser.add_argument('--pool_every_timestep', default=1, type=bool_flag)

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=1024, type=int)

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=64, type=int)
parser.add_argument('--d_learning_rate', default=1e-4, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
parser.add_argument('--l2_loss_weight', default=1.0, type=float)
parser.add_argument('--best_k', default=6, type=int)

# Output
parser.add_argument('--output_dir', default='ckpts')
parser.add_argument('--print_every', default=5, type=int)
parser.add_argument('--checkpoint_every', default=100, type=int)
parser.add_argument('--checkpoint_name', default='baseline')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)

# HighD dataset options
parser.add_argument('--use_highd', default=0, type=bool_flag,
                    help='Use HighD mmap dataset instead of ETH/UCY text files.')
parser.add_argument('--highd_mmap_path', default='data/highD/mmap', type=str,
                    help='Path to the mmap directory produced by preprocess.py.')
parser.add_argument('--highd_split_dir', default='', type=str,
                    help='Directory containing train/val/test_indices.npy produced by split.py. '
                         'When set, pre-computed splits are used instead of highd_val_ratio.')
parser.add_argument('--highd_val_ratio', default=0.1, type=float,
                    help='Fraction of recordings to use as validation '
                         '(only when --highd_split_dir is not set).')
parser.add_argument('--feature_mode', default=None,
                    choices=['baseline', 'dimI'],
                    help='HighD neighbor feature mode: baseline=[0:6], dimI=[0:6]+dim(8)+I(9).')
parser.add_argument('--use_I', default=0, type=bool_flag,
                    help=argparse.SUPPRESS)
parser.add_argument('--use_Iy', default=0, type=bool_flag,
                    help=argparse.SUPPRESS)
parser.add_argument('--use_dim', default=0, type=bool_flag,
                    help=argparse.SUPPRESS)
parser.add_argument('--seed', default=None, type=int,
                    help='Global random seed for reproducibility.')


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def main(args):
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Global seed set to %d", args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    long_dtype, float_dtype = get_dtypes(args)

    if args.feature_mode is None:
        if args.use_dim and args.use_I:
            args.feature_mode = 'dimI'
        elif not args.use_dim and not args.use_I and not args.use_Iy:
            args.feature_mode = 'baseline'

    if args.use_highd:
        if args.highd_split_dir:
            logger.info("Using pre-computed splits from %s", args.highd_split_dir)

        # Derive obs_len / pred_len from the mmap before constructing datasets
        import numpy as np
        from pathlib import Path
        _x = np.load(str(Path(args.highd_mmap_path) / 'x_ego.npy'), mmap_mode='r')
        _y = np.load(str(Path(args.highd_mmap_path) / 'y.npy'),     mmap_mode='r')
        args.obs_len  = _x.shape[1]
        args.pred_len = _y.shape[1]
        logger.info("HighD mmap: obs_len=%d  pred_len=%d", args.obs_len, args.pred_len)

        logger.info("Initializing HighD train dataset from %s", args.highd_mmap_path)
        train_dset, train_loader = highd_data_loader(
            args, args.highd_mmap_path, split='train')
        logger.info("Initializing HighD val dataset")
        _, val_loader = highd_data_loader(
            args, args.highd_mmap_path, split='val')

        # Recommend highd_pool when the user hasn't changed pooling_type
        if args.pooling_type == 'pool_net':
            logger.warning(
                "Using --pooling_type pool_net with HighD data provides no "
                "social context (only 1 ego per scene). "
                "Consider --pooling_type highd_pool to use neighbor features."
            )
    else:
        train_path = get_dset_path(args.dataset_name, 'train')
        val_path   = get_dset_path(args.dataset_name, 'val')
        logger.info("Initializing train dataset")
        train_dset, train_loader = data_loader(args, train_path)
        logger.info("Initializing val dataset")
        _, val_loader = data_loader(args, val_path)

    iterations_per_epoch = len(train_dset) / args.batch_size / args.d_steps
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    nb_feat_dim = highd_nb_feat_dim(
        getattr(args, 'feature_mode', None),
        use_I=getattr(args, 'use_I', False),
        use_Iy=getattr(args, 'use_Iy', False),
        use_dim=getattr(args, 'use_dim', False),
    )
    args.nb_feat_dim = nb_feat_dim
    logger.info("HighD feature_mode=%s  nb_feat_dim=%d",
                getattr(args, 'feature_mode', None), nb_feat_dim)

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
        nb_K=8)

    generator.apply(init_weights)
    generator.type(float_dtype).train()

    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type)

    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=args.d_learning_rate
    )

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path, weights_only=False)
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'norm_d': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
            'g_best_state': None,
            'd_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'd_best_state_nl': None,
            'best_t_nl': None,
        }
    os.makedirs(args.output_dir, exist_ok=True)
    start_epoch  = epoch
    end_epoch    = start_epoch + args.num_epochs
    best_ade     = float('inf')
    best_path    = os.path.join(args.output_dir, '%s_best.pt' % args.checkpoint_name)
    last_path    = os.path.join(args.output_dir, '%s_with_model.pt' % args.checkpoint_name)

    print(f'\n====== Train ======')
    print(f'  Epochs={args.num_epochs}  bs={args.batch_size}'
          f'  lr_g={args.g_learning_rate}  lr_d={args.d_learning_rate}'
          f'  ckpt={args.output_dir}')

    for ep in range(start_epoch + 1, end_epoch + 1):
        gc.collect()
        rel_ep = ep - start_epoch
        print(f'\n====== Epoch {rel_ep}/{args.num_epochs} ======')

        d_losses_ep, g_losses_ep = [], []
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        losses_d = losses_g = {}
        steps_per_iter = args.d_steps + args.g_steps
        iters_per_epoch = len(train_loader) // steps_per_iter

        pbar = tqdm(total=iters_per_epoch, desc='  train', leave=True, ncols=100)
        for batch in train_loader:
            if d_steps_left > 0:
                losses_d = discriminator_step(args, batch, generator,
                                              discriminator, d_loss_fn,
                                              optimizer_d)
                checkpoint['norm_d'].append(
                    get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
            elif g_steps_left > 0:
                losses_g = generator_step(args, batch, generator,
                                          discriminator, g_loss_fn,
                                          optimizer_g)
                checkpoint['norm_g'].append(
                    get_total_norm(generator.parameters()))
                g_steps_left -= 1

            if d_steps_left > 0 or g_steps_left > 0:
                continue

            # full iteration complete
            d_losses_ep.append(losses_d.get('D_total_loss', 0.0))
            g_losses_ep.append(losses_g.get('G_total_loss', 0.0))
            for k, v in losses_d.items():
                checkpoint['D_losses'][k].append(v)
            for k, v in losses_g.items():
                checkpoint['G_losses'][k].append(v)

            t += 1
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            pbar.set_postfix(
                D=f"{losses_d.get('D_total_loss', 0):.3f}",
                G=f"{losses_g.get('G_total_loss', 0):.3f}",
            )
            pbar.update(1)

        pbar.close()
        avg_d = sum(d_losses_ep) / len(d_losses_ep) if d_losses_ep else 0.0
        avg_g = sum(g_losses_ep) / len(g_losses_ep) if g_losses_ep else 0.0

        # Validation
        metrics_val = check_accuracy(
            args, val_loader, generator, discriminator, d_loss_fn
        )
        for k, v in metrics_val.items():
            checkpoint['metrics_val'][k].append(v)

        print(
            f'  D_loss={avg_d:.4f}  G_loss={avg_g:.4f} | '
            f'Val ADE={metrics_val["ade"]:.3f}  FDE={metrics_val["fde"]:.3f}'
        )

        # Save checkpoint
        checkpoint['counters']['t'] = t
        checkpoint['counters']['epoch'] = ep
        checkpoint['g_state'] = generator.state_dict()
        checkpoint['g_optim_state'] = optimizer_g.state_dict()
        checkpoint['d_state'] = discriminator.state_dict()
        checkpoint['d_optim_state'] = optimizer_d.state_dict()
        torch.save(checkpoint, last_path)

        if metrics_val['ade'] < best_ade:
            best_ade = metrics_val['ade']
            checkpoint['best_t'] = t
            checkpoint['g_best_state'] = generator.state_dict()
            checkpoint['d_best_state'] = discriminator.state_dict()
            torch.save(checkpoint, best_path)
            print(f'  best ckpt saved -> {best_path}  (val_ade={best_ade:.4f})')

        if ep % args.checkpoint_every == 0:
            snap_path = os.path.join(
                args.output_dir, '%s_ep%d.pt' % (args.checkpoint_name, ep))
            torch.save(checkpoint, snap_path)
            print(f'  snapshot -> {snap_path}')

    print(f'\n[DONE] Training finished.  Best val_ade: {best_ade:.4f}')


def _unpack_batch(args, batch):
    """
    Unpack a batch from either the original collate or seq_collate_highd.

    Returns
    -------
    obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
    non_linear_ped, loss_mask, seq_start_end,
    nb_feats (or None), nb_mask (or None)
    """
    batch = [tensor.cuda() for tensor in batch]
    if args.use_highd:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
         non_linear_ped, loss_mask, seq_start_end,
         nb_feats, nb_mask) = batch
    else:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
         non_linear_ped, loss_mask, seq_start_end) = batch
        nb_feats = nb_mask = None
    return (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
            non_linear_ped, loss_mask, seq_start_end, nb_feats, nb_mask)


def _generator_forward(args, generator, obs_traj, obs_traj_rel,
                        seq_start_end, nb_feats=None, nb_mask=None):
    return generator(obs_traj, obs_traj_rel, seq_start_end, nb_feats, nb_mask)


def discriminator_step(
    args, batch, generator, discriminator, d_loss_fn, optimizer_d
):
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
     non_linear_ped, loss_mask, seq_start_end,
     nb_feats, nb_mask) = _unpack_batch(args, batch)
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    generator_out = _generator_forward(
        args, generator, obs_traj, obs_traj_rel, seq_start_end, nb_feats, nb_mask)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 args.clipping_threshold_d)
    optimizer_d.step()

    return losses


def generator_step(
    args, batch, generator, discriminator, g_loss_fn, optimizer_g
):
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
     non_linear_ped, loss_mask, seq_start_end,
     nb_feats, nb_mask) = _unpack_batch(args, batch)
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, args.obs_len:]

    for _ in range(args.best_k):
        generator_out = _generator_forward(
            args, generator, obs_traj, obs_traj_rel, seq_start_end, nb_feats, nb_mask)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode='raw'))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), args.clipping_threshold_g
        )
    optimizer_g.step()

    return losses


def check_accuracy(
    args, loader, generator, discriminator, d_loss_fn, limit=False
):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end,
             nb_feats, nb_mask) = _unpack_batch(args, batch)
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            pred_traj_fake_rel = _generator_forward(
                args, generator, obs_traj, obs_traj_rel, seq_start_end,
                nb_feats, nb_mask)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()
    return metrics


def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
    loss_mask
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_ped
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
    )
    return fde, fde_l, fde_nl


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
