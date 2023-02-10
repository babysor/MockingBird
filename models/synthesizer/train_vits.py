import os
from loguru import logger
import torch
import glob
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from utils.audio_utils import mel_spectrogram, spec_to_mel
from utils.loss import feature_loss, generator_loss, discriminator_loss, kl_loss
from utils.util import slice_segments, clip_grad_value_
from models.synthesizer.vits_dataset import (
    VitsDataset,
    VitsDatasetCollate,
    DistributedBucketSampler
)
from models.synthesizer.models.vits import (
    Vits,
    MultiPeriodDiscriminator,
)
from models.synthesizer.utils.symbols import symbols
from models.synthesizer.utils.plot import plot_spectrogram_to_numpy, plot_alignment_to_numpy
from pathlib import Path
from utils.hparams import HParams
import torch.multiprocessing as mp
import argparse

# torch.backends.cudnn.benchmark = True
global_step = 0


def new_train():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_dir", type=str, default="../audiodata/SV2TTS/synthesizer", help= \
        "Path to the synthesizer directory that contains the ground truth mel spectrograms, "
        "the wavs, the emos and the embeds.")
    parser.add_argument("-m", "--model_dir", type=str, default="data/ckpt/synthesizer/vits2", help=\
        "Path to the output directory that will contain the saved model weights and the logs.")
    parser.add_argument('--ckptG', type=str, required=False,
                      help='original VITS G checkpoint path')
    parser.add_argument('--ckptD', type=str, required=False,
                      help='original VITS D checkpoint path')
    args, _ = parser.parse_known_args()

    datasets_root = Path(args.syn_dir)
    hparams= HParams(
        model_dir = args.model_dir,
    )
    hparams.loadJson(Path(hparams.model_dir).joinpath("config.json"))
    hparams.data["training_files"] = str(datasets_root.joinpath("train.txt"))
    hparams.data["validation_files"] = str(datasets_root.joinpath("train.txt"))
    hparams.data["datasets_root"] = str(datasets_root)
    hparams.ckptG = args.ckptG
    hparams.ckptD = args.ckptD
    n_gpus = torch.cuda.device_count()
    # for spawn
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8899'
    # mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hparams))
    run(0, 1, hparams)


def load_checkpoint(checkpoint_path, model, optimizer=None, is_old=False, epochs=10000):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  if optimizer is not None:
    if not is_old:
      optimizer.load_state_dict(checkpoint_dict['optimizer'])
    else:
      new_opt_dict = optimizer.state_dict()
      new_opt_dict_params = new_opt_dict['param_groups'][0]['params']
      new_opt_dict['param_groups'] = checkpoint_dict['optimizer']['param_groups']
      new_opt_dict['param_groups'][0]['params'] = new_opt_dict_params
      optimizer.load_state_dict(new_opt_dict)
  saved_state_dict = checkpoint_dict['model']
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict= {}
  for k, v in state_dict.items():
    try:
      new_state_dict[k] = saved_state_dict[k]
    except:
        if k == 'step':
            new_state_dict[k] = iteration * epochs
        else:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v

  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict, strict=False)
  else:
    model.load_state_dict(new_state_dict, strict=False)
  logger.info("Loaded checkpoint '{}' (iteration {})" .format(
    checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
  logger.info("Saving model and optimizer state at iteration {} to {}".format(
    iteration, checkpoint_path))
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer.state_dict(),
              'learning_rate': learning_rate}, checkpoint_path)

def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x

def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger.info(hps)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(backend='gloo', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    train_dataset = VitsDataset(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    collate_fn = VitsDatasetCollate()
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
                              collate_fn=collate_fn, batch_sampler=train_sampler)
    if rank == 0:
        eval_dataset = VitsDataset(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False,
                                 batch_size=hps.train.batch_size, pin_memory=True,
                                 drop_last=False, collate_fn=collate_fn)

    net_g = Vits(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])
    ckptG = hps.ckptG
    ckptD = hps.ckptD
    try:
        if ckptG is not None:
            _, _, _, epoch_str = load_checkpoint(ckptG, net_g, optim_g, is_old=True)
            print("加载原版VITS模型G记录点成功")
        else:
            _, _, _, epoch_str = load_checkpoint(latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                                                   optim_g, epochs=hps.train.epochs)
        if ckptD is not None:
            _, _, _, epoch_str = load_checkpoint(ckptG, net_g, optim_g, is_old=True)
            print("加载原版VITS模型D记录点成功")
        else:
            _, _, _, epoch_str = load_checkpoint(latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
                                                   optim_d, epochs=hps.train.epochs)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0
    if ckptG is not None or ckptD is not None:
        epoch_str = 1
        global_step = 0
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, None], None, None)
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers
    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, emo) in enumerate(train_loader):
        # logger.info(f'====> Step: 1 {batch_idx}')
        x, x_lengths = x.cuda(rank), x_lengths.cuda(rank)
        spec, spec_lengths = spec.cuda(rank), spec_lengths.cuda(rank)
        y, y_lengths = y.cuda(rank), y_lengths.cuda(rank)
        speakers = speakers.cuda(rank)
        emo = emo.cuda(rank)
        # logger.info(f'====> Step: 1.0 {batch_idx}')
        with autocast(enabled=hps.train.fp16_run):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, speakers, emo)
            # logger.info(f'====> Step: 1.1 {batch_idx}')
            mel = spec_to_mel(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)
            y_mel = slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            y = slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice
            # logger.info(f'====> Step: 1.3 {batch_idx}')
            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all.float()).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()
        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr,
                               "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                scalar_dict.update(
                    {"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})

                scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = {
                    "slice/mel_org": plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    "all/attn": plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy())
                }
                summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict)

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
        global_step += 1

    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, emo) in enumerate(eval_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)
            speakers = speakers.cuda(0)
            emo = emo.cuda(0)
            # remove else
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            speakers = speakers[:1]
            emo = emo[:1]
            break
        y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, speakers, emo, max_len=1000)
        # y_hat, attn, mask, *_ = generator.infer(x, x_lengths, speakers, emo, max_len=1000) # for non DistributedDataParallel object
        
        y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

        mel = spec_to_mel(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax)
        y_hat_mel = mel_spectrogram(
            y_hat.squeeze(1).float(),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
    image_dict = {
        "gen/mel": plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
        "gen/audio": y_hat[0, :, :y_hat_lengths[0]]
    }
    if global_step == 0:
        image_dict.update({"gt/mel": plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
        audio_dict.update({"gt/audio": y[0, :, :y_lengths[0]]})

    summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)

