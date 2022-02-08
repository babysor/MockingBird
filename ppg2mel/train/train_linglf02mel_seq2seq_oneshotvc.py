import os, sys
# sys.path.append('/home/shaunxliu/projects/nnsp')
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from torch.utils.data import DataLoader
import numpy as np
from .solver import BaseSolver
from utils.data_load import OneshotVcDataset, MultiSpkVcCollate
# from src.rnn_ppg2mel import BiRnnPpg2MelModel
# from src.mel_decoder_mol_encAddlf0 import MelDecoderMOL
from .loss import MaskedMSELoss
from .optim import Optimizer
from utils.util import human_format
from ppg2mel import MelDecoderMOLv2


class Solver(BaseSolver):
    """Customized Solver."""
    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        self.num_att_plots = 5
        self.att_ws_dir = f"{self.logdir}/att_ws"
        os.makedirs(self.att_ws_dir, exist_ok=True)
        self.best_loss = np.inf

    def fetch_data(self, data):
        """Move data to device"""
        data = [i.to(self.device) for i in data]
        return data

    def load_data(self):
        """ Load data for training/validation/plotting."""
        train_dataset = OneshotVcDataset(
            meta_file=self.config.data.train_fid_list,
            vctk_ppg_dir=self.config.data.vctk_ppg_dir,
            libri_ppg_dir=self.config.data.libri_ppg_dir,
            vctk_f0_dir=self.config.data.vctk_f0_dir,
            libri_f0_dir=self.config.data.libri_f0_dir,
            vctk_wav_dir=self.config.data.vctk_wav_dir,
            libri_wav_dir=self.config.data.libri_wav_dir,
            vctk_spk_dvec_dir=self.config.data.vctk_spk_dvec_dir,
            libri_spk_dvec_dir=self.config.data.libri_spk_dvec_dir,
            ppg_file_ext=self.config.data.ppg_file_ext,
            min_max_norm_mel=self.config.data.min_max_norm_mel,
            mel_min=self.config.data.mel_min,
            mel_max=self.config.data.mel_max,
        )
        dev_dataset = OneshotVcDataset(
            meta_file=self.config.data.dev_fid_list,
            vctk_ppg_dir=self.config.data.vctk_ppg_dir,
            libri_ppg_dir=self.config.data.libri_ppg_dir,
            vctk_f0_dir=self.config.data.vctk_f0_dir,
            libri_f0_dir=self.config.data.libri_f0_dir,
            vctk_wav_dir=self.config.data.vctk_wav_dir,
            libri_wav_dir=self.config.data.libri_wav_dir,
            vctk_spk_dvec_dir=self.config.data.vctk_spk_dvec_dir,
            libri_spk_dvec_dir=self.config.data.libri_spk_dvec_dir,
            ppg_file_ext=self.config.data.ppg_file_ext,
            min_max_norm_mel=self.config.data.min_max_norm_mel,
            mel_min=self.config.data.mel_min,
            mel_max=self.config.data.mel_max,
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            num_workers=self.paras.njobs,
            shuffle=True,
            batch_size=self.config.hparas.batch_size,
            pin_memory=False,
            drop_last=True,
            collate_fn=MultiSpkVcCollate(self.config.model.frames_per_step,
                                        use_spk_dvec=True),
        )
        self.dev_dataloader = DataLoader(
            dev_dataset,
            num_workers=self.paras.njobs,
            shuffle=False,
            batch_size=self.config.hparas.batch_size,
            pin_memory=False,
            drop_last=False,
            collate_fn=MultiSpkVcCollate(self.config.model.frames_per_step,
                                         use_spk_dvec=True),
        )
        self.plot_dataloader = DataLoader(
            dev_dataset,
            num_workers=self.paras.njobs,
            shuffle=False,
            batch_size=1,
            pin_memory=False,
            drop_last=False,
            collate_fn=MultiSpkVcCollate(self.config.model.frames_per_step,
                                         use_spk_dvec=True,
                                         give_uttids=True),
        )
        msg = "Have prepared training set and dev set."
        self.verbose(msg)
    
    def load_pretrained_params(self):
        print("Load pretrained model from: ", self.config.data.pretrain_model_file)
        ignore_layer_prefixes = ["speaker_embedding_table"]
        pretrain_model_file = self.config.data.pretrain_model_file
        pretrain_ckpt = torch.load(
            pretrain_model_file, map_location=self.device
        )["model"]
        model_dict = self.model.state_dict()
        print(self.model)
        
        # 1. filter out unnecessrary keys
        for prefix in ignore_layer_prefixes:
            pretrain_ckpt = {k : v 
                             for k, v in pretrain_ckpt.items() if not k.startswith(prefix) 
                            }
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrain_ckpt)

        # 3. load the new state dict
        self.model.load_state_dict(model_dict)

    def set_model(self):
        """Setup model and optimizer"""
        # Model
        print("[INFO] Model name: ", self.config["model_name"])
        self.model = MelDecoderMOLv2(
            **self.config["model"]
        ).to(self.device)
        # self.load_pretrained_params()

        # model_params = [{'params': self.model.spk_embedding.weight}]
        model_params = [{'params': self.model.parameters()}]
        
        # Loss criterion
        self.loss_criterion = MaskedMSELoss(self.config.model.frames_per_step)

        # Optimizer
        self.optimizer = Optimizer(model_params, **self.config["hparas"])
        self.verbose(self.optimizer.create_msg())

        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt()

    def exec(self):
        self.verbose("Total training steps {}.".format(
            human_format(self.max_step)))

        mel_loss = None
        n_epochs = 0
        # Set as current time
        self.timer.set()
        
        while self.step < self.max_step:
            for data in self.train_dataloader:
                # Pre-step: updata lr_rate and do zero_grad
                lr_rate = self.optimizer.pre_step(self.step)
                total_loss = 0
                # data to device
                ppgs, lf0_uvs, mels, in_lengths, \
                    out_lengths, spk_ids, stop_tokens = self.fetch_data(data)
                self.timer.cnt("rd")
                mel_outputs, mel_outputs_postnet, predicted_stop = self.model(
                    ppgs,
                    in_lengths,
                    mels,
                    out_lengths,
                    lf0_uvs,
                    spk_ids
                ) 
                mel_loss, stop_loss = self.loss_criterion(
                    mel_outputs,
                    mel_outputs_postnet,
                    mels,
                    out_lengths,
                    stop_tokens,
                    predicted_stop
                )
                loss = mel_loss + stop_loss

                self.timer.cnt("fw")

                # Back-prop
                grad_norm = self.backward(loss)
                self.step += 1

                # Logger
                if (self.step == 1) or (self.step % self.PROGRESS_STEP == 0):
                    self.progress("Tr|loss:{:.4f},mel-loss:{:.4f},stop-loss:{:.4f}|Grad.Norm-{:.2f}|{}"
                                  .format(loss.cpu().item(), mel_loss.cpu().item(),
                                    stop_loss.cpu().item(), grad_norm, self.timer.show()))
                    self.write_log('loss', {'tr/loss': loss,
                                            'tr/mel-loss': mel_loss,
                                            'tr/stop-loss': stop_loss})

                # Validation
                if (self.step == 1) or (self.step % self.valid_step == 0):
                    self.validate()

                # End of step
                # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                torch.cuda.empty_cache()
                self.timer.set()
                if self.step > self.max_step:
                    break
            n_epochs += 1
        self.log.close()

    def validate(self):
        self.model.eval()
        dev_loss, dev_mel_loss, dev_stop_loss = 0.0, 0.0, 0.0

        for i, data in enumerate(self.dev_dataloader):
            self.progress('Valid step - {}/{}'.format(i+1, len(self.dev_dataloader)))
            # Fetch data
            ppgs, lf0_uvs, mels, in_lengths, \
                out_lengths, spk_ids, stop_tokens = self.fetch_data(data)
            with torch.no_grad():
                mel_outputs, mel_outputs_postnet, predicted_stop = self.model(
                    ppgs,
                    in_lengths,
                    mels,
                    out_lengths,
                    lf0_uvs,
                    spk_ids
                ) 
                mel_loss, stop_loss = self.loss_criterion(
                    mel_outputs,
                    mel_outputs_postnet,
                    mels,
                    out_lengths,
                    stop_tokens,
                    predicted_stop
                )
                loss = mel_loss + stop_loss

                dev_loss += loss.cpu().item()
                dev_mel_loss += mel_loss.cpu().item()
                dev_stop_loss += stop_loss.cpu().item()

        dev_loss = dev_loss / (i + 1)
        dev_mel_loss = dev_mel_loss / (i + 1)
        dev_stop_loss = dev_stop_loss / (i + 1)
        self.save_checkpoint(f'step_{self.step}.pth', 'loss', dev_loss, show_msg=False)
        if dev_loss < self.best_loss:
            self.best_loss = dev_loss
            self.save_checkpoint(f'best_loss_step_{self.step}.pth', 'loss', dev_loss)
        self.write_log('loss', {'dv/loss': dev_loss,
                                'dv/mel-loss': dev_mel_loss,
                                'dv/stop-loss': dev_stop_loss})

        # plot attention
        for i, data in enumerate(self.plot_dataloader):
            if i == self.num_att_plots:
                break
            # Fetch data
            ppgs, lf0_uvs, mels, in_lengths, \
                out_lengths, spk_ids, stop_tokens = self.fetch_data(data[:-1])
            fid = data[-1][0]
            with torch.no_grad():
                _, _, _, att_ws = self.model(
                    ppgs,
                    in_lengths,
                    mels,
                    out_lengths,
                    lf0_uvs,
                    spk_ids,
                    output_att_ws=True
                )
                att_ws = att_ws.squeeze(0).cpu().numpy()
                att_ws = att_ws[None]
                w, h = plt.figaspect(1.0 / len(att_ws))
                fig = plt.Figure(figsize=(w * 1.3, h * 1.3))
                axes = fig.subplots(1, len(att_ws))
                if len(att_ws) == 1:
                    axes = [axes]

                for ax, aw in zip(axes, att_ws):
                    ax.imshow(aw.astype(np.float32), aspect="auto")
                    ax.set_title(f"{fid}")
                    ax.set_xlabel("Input")
                    ax.set_ylabel("Output")
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                fig_name = f"{self.att_ws_dir}/{fid}_step{self.step}.png"
                fig.savefig(fig_name)
                
        # Resume training
        self.model.train()

