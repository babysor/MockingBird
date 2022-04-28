import matplotlib.pyplot as plt
import numpy
from scipy.fftpack import dct
from PyQt5.QtGui import QPalette, QBrush, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QStringListModel
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from encoder.inference import plot_embedding_as_heatmap
from toolbox.utterance import Utterance
from pathlib import Path
from typing import List, Set
import sounddevice as sd
import soundfile as sf
import numpy as np
# from sklearn.manifold import TSNE         # You can try with TSNE if you like, I prefer UMAP 
from time import sleep
import umap
import sys
from warnings import filterwarnings, warn



filterwarnings("ignore")





colormap = np.array([
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [97, 142, 151],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
    [76, 255, 0],
], dtype=np.float) / 255 

default_text = \
    "请输入需要克隆的语音文本！"


   
class UI(QDialog):
    min_umap_points = 4
    max_log_lines = 5
    max_saved_utterances = 20
    
    def draw_utterance(self, utterance: Utterance, which):
        self.draw_spec(utterance.spec, which)
        self.draw_embed(utterance.embed, utterance.name, which)






    def draw_embed(self, embed, name, which):
        embed_ax, _ = self.current_ax if which == "current" else self.gen_ax
        embed_ax.figure.suptitle("" if embed is None else name)
        
        ## Embedding
        # Clear the plot
        if len(embed_ax.images) > 0:
            embed_ax.images[0].colorbar.remove()
        embed_ax.clear()
        
        # Draw the embed
        if embed is not None:
            plot_embedding_as_heatmap(embed, embed_ax)
            embed_ax.set_title("embedding")
        embed_ax.set_aspect("equal", "datalim")
        embed_ax.set_xticks([])
        embed_ax.set_yticks([])
        embed_ax.figure.canvas.draw()

    def draw_spec(self, spec, which):
        _, spec_ax = self.current_ax if which == "current" else self.gen_ax

        ## Spectrogram
        # Draw the spectrogram
        spec_ax.clear()
        if spec is not None:
            im = spec_ax.imshow(spec, aspect="auto", interpolation="none")
            # spec_ax.figure.colorbar(mappable=im, shrink=0.65, orientation="horizontal", 
            # spec_ax=spec_ax)
            spec_ax.set_title("mel spectrogram")
    
        spec_ax.set_xticks([])
        spec_ax.set_yticks([])
        spec_ax.figure.canvas.draw()
        if which != "current":
            self.vocode_button.setDisabled(spec is None)

    def draw_umap_projections(self, utterances: Set[Utterance]):
        self.umap_ax.clear()

        speakers = np.unique([u.speaker_name for u in utterances])
        colors = {speaker_name: colormap[i] for i, speaker_name in enumerate(speakers)}
        embeds = [u.embed for u in utterances]

        # Display a message if there aren't enough points
        if len(utterances) < self.min_umap_points:
            self.umap_ax.text(.5, .5, "umap:\nAdd %d more points to\ngenerate the projections" %
                              (self.min_umap_points - len(utterances)), 
                              horizontalalignment='center', fontsize=15)
            self.umap_ax.set_title("")
            
        # Compute the projections
        else:
            if not self.umap_hot:
                self.log(
                    "Drawing UMAP projections for the first time, this will take a few seconds.")
                self.umap_hot = True
            
            reducer = umap.UMAP(int(np.ceil(np.sqrt(len(embeds)))), metric="cosine")
            # reducer = TSNE()
            projections = reducer.fit_transform(embeds)
            
            speakers_done = set()
            for projection, utterance in zip(projections, utterances):
                color = colors[utterance.speaker_name]
                mark = "x" if "_gen_" in utterance.name else "o"
                label = None if utterance.speaker_name in speakers_done else utterance.speaker_name
                speakers_done.add(utterance.speaker_name)
                self.umap_ax.scatter(projection[0], projection[1], c=[color], marker=mark,
                                     label=label)
            # self.umap_ax.set_title("UMAP projections")
            self.umap_ax.legend(prop={'size': 10})

        # Draw the plot
        self.umap_ax.set_aspect("equal", "datalim")
        self.umap_ax.set_xticks([])
        self.umap_ax.set_yticks([])
        self.umap_ax.figure.canvas.draw()

    def save_audio_file(self, wav, sample_rate):        
        dialog = QFileDialog()
        dialog.setDefaultSuffix(".wav")
        fpath, _ = dialog.getSaveFileName(
            parent=self,
            caption="Select a path to save the audio file",
            filter="Audio Files (*.flac *.wav)"
        )
        if fpath:
            #Default format is wav
            if Path(fpath).suffix == "":
                fpath += ".wav"
            sf.write(fpath, wav, sample_rate)

    def setup_audio_devices(self, sample_rate):
        input_devices = []
        output_devices = []
        for device in sd.query_devices():
            # Check if valid input
            try:
                sd.check_input_settings(device=device["name"], samplerate=sample_rate)
                input_devices.append(device["name"])
            except:
                pass

            # Check if valid output
            try:
                sd.check_output_settings(device=device["name"], samplerate=sample_rate)
                output_devices.append(device["name"])
            except Exception as e:
                # Log a warning only if the device is not an input
                if not device["name"] in input_devices:
                    warn("Unsupported output device %s for the sample rate: %d \nError: %s" % (device["name"], sample_rate, str(e)))

        if len(input_devices) == 0:
            self.log("No audio input device detected. Recording may not work.")
            self.audio_in_device = None
        else:
            self.audio_in_device = input_devices[0]

        if len(output_devices) == 0:
            self.log("No supported output audio devices were found! Audio output may not work.")
            self.audio_out_devices_cb.addItems(["None"])
            self.audio_out_devices_cb.setDisabled(True)
        else:
            self.audio_out_devices_cb.clear()
            self.audio_out_devices_cb.addItems(output_devices)
            self.audio_out_devices_cb.currentTextChanged.connect(self.set_audio_device)

        self.set_audio_device()

    def set_audio_device(self):
        
        output_device = self.audio_out_devices_cb.currentText()
        if output_device == "None":
            output_device = None

        # If None, sounddevice queries portaudio
        sd.default.device = (self.audio_in_device, output_device)
    
    def play(self, wav, sample_rate):
        try:
            sd.stop()
            sd.play(wav, sample_rate)
        except Exception as e:
            print(e)
            self.log("Error in audio playback. Try selecting a different audio output device.")
            self.log("Your device must be connected before you start the toolbox.")
        
    def stop(self):
        sd.stop()

    def record_one(self, sample_rate, duration):
        self.record_button.setText("Recording...")
        self.record_button.setDisabled(True)
        
        self.log("Recording %d seconds of audio" % duration)
        sd.stop()
        try:
            wav = sd.rec(duration * sample_rate, sample_rate, 1)
        except Exception as e:
            print(e)
            self.log("Could not record anything. Is your recording device enabled?")
            self.log("Your device must be connected before you start the toolbox.")
            return None
        
        for i in np.arange(0, duration, 0.1):
            self.set_loading(i, duration)
            sleep(0.1)
        self.set_loading(duration, duration)
        sd.wait()
        
        self.log("Done recording.")
        self.record_button.setText("Record")
        self.record_button.setDisabled(False)
        
        return wav.squeeze()



    #添加source_mfcc分析函数
    def plot_mfcc(self, wav, sample_rate):

        signal = wav
        print(sample_rate, len(signal))
        # 读取前3.5s 的数据
        signal = signal[0:int(3.5 * sample_rate)]
        print(signal)

        # 预先处理
        pre_emphasis = 0.97
        emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        frame_size = 0.025
        frame_stride = 0.1
        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))

        pad_signal_length = num_frames * frame_step + frame_length
        z = numpy.zeros((pad_signal_length - signal_length))
        pad_signal = numpy.append(emphasized_signal, z)

        indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(
            numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

        frames = pad_signal[numpy.mat(indices).astype(numpy.int32, copy=False)]

        # 加上汉明窗
        frames *= numpy.hamming(frame_length)
        # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **

        # 傅立叶变换和功率谱
        NFFT = 512
        mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        # print(mag_frames.shape)
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

        low_freq_mel = 0
        # 将频率转换为Mel
        nfilt = 40
        high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))
        mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz

        bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))

        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = numpy.dot(pow_frames, fbank.T)
        filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * numpy.log10(filter_banks)  # dB

        # 所得到的倒谱系数2-13被保留，其余的被丢弃
        num_ceps = 12
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]
        (nframes, ncoeff) = mfcc.shape

        n = numpy.arange(ncoeff)
        cep_lifter = 22
        lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
        mfcc *= lift  # *

        # filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
        mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
        print(mfcc.shape)

        # 创建新的figure
        fig10 = plt.figure(figsize=(16,8))

        # 绘制1x2两行两列共四个图，编号从1开始
        ax = fig10.add_subplot(121)
        plt.plot(mfcc)

        ax = fig10.add_subplot(122)
        # 平均归一化MFCC
        mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
        plt.imshow(numpy.flipud(mfcc.T), cmap=plt.cm.jet, aspect=0.2,
                   extent=[0, mfcc.shape[0], 0, mfcc.shape[1]])  # 热力图
        #将figure保存为png并显示在新创建的子窗口上
        plt.savefig("fmcc_source.png")
        dialog_fault = QDialog()
        dialog_fault.setWindowTitle("源音频MFCC特征图及MFCC平均归一化热图")  # 设置窗口名
        pic = QPixmap("fmcc_source.png")
        label_pic = QLabel("show", dialog_fault)
        label_pic.setPixmap(pic)
        label_pic.setGeometry(0,0,1500,800)
        dialog_fault.exec_()




    @property        
    def current_dataset_name(self):
        return self.dataset_box.currentText()

    @property
    def current_speaker_name(self):
        return self.speaker_box.currentText()
    
    @property
    def current_utterance_name(self):
        return self.utterance_box.currentText()
    
    def browse_file(self):
        fpath = QFileDialog().getOpenFileName(
            parent=self,
            caption="Select an audio file",
            filter="Audio Files (*.mp3 *.flac *.wav *.m4a)"
        )
        return Path(fpath[0]) if fpath[0] != "" else ""
    
    @staticmethod
    def repopulate_box(box, items, random=False):
        """
        Resets a box and adds a list of items. Pass a list of (item, data) pairs instead to join 
        data to the items
        """
        box.blockSignals(True)
        box.clear()
        for item in items:
            item = list(item) if isinstance(item, tuple) else [item]
            box.addItem(str(item[0]), *item[1:])
        if len(items) > 0:
            box.setCurrentIndex(np.random.randint(len(items)) if random else 0)
        box.setDisabled(len(items) == 0)
        box.blockSignals(False)
    
    def populate_browser(self, datasets_root: Path, recognized_datasets: List, level: int,
                         random=True):
        # Select a random dataset
        if level <= 0:
            if datasets_root is not None:
                datasets = [datasets_root.joinpath(d) for d in recognized_datasets]
                datasets = [d.relative_to(datasets_root) for d in datasets if d.exists()]
                self.browser_load_button.setDisabled(len(datasets) == 0)
            if datasets_root is None or len(datasets) == 0:
                msg = "Tip: Please " + (" select the voice to be cloned" \
                    if datasets_root is None else "o not have any of the recognized datasets" \
                                                  " in %s" % datasets_root) 
                self.log(msg)
                msg += ".\nThe recognized datasets are:\n\t%s\nFeel free to add your own. You " \
                       "can still use the toolbox by recording samples yourself." % \
                       ("\n\t".join(recognized_datasets))
                print(msg, file=sys.stderr)
                
                self.random_utterance_button.setDisabled(True)
                self.random_speaker_button.setDisabled(True)
                self.random_dataset_button.setDisabled(True)
                self.utterance_box.setDisabled(True)
                self.speaker_box.setDisabled(True)
                self.dataset_box.setDisabled(True)
                self.browser_load_button.setDisabled(True)
                self.auto_next_checkbox.setDisabled(True)
                return 
            self.repopulate_box(self.dataset_box, datasets, random)
    
        # Select a random speaker
        if level <= 1:
            speakers_root = datasets_root.joinpath(self.current_dataset_name)
            speaker_names = [d.stem for d in speakers_root.glob("*") if d.is_dir()]
            self.repopulate_box(self.speaker_box, speaker_names, random)
    
        # Select a random utterance
        if level <= 2:
            utterances_root = datasets_root.joinpath(
                self.current_dataset_name, 
                self.current_speaker_name
            )
            utterances = []
            for extension in ['mp3', 'flac', 'wav', 'm4a']:
                utterances.extend(Path(utterances_root).glob("**/*.%s" % extension))
            utterances = [fpath.relative_to(utterances_root) for fpath in utterances]
            self.repopulate_box(self.utterance_box, utterances, random)
            
    def browser_select_next(self):
        index = (self.utterance_box.currentIndex() + 1) % len(self.utterance_box)
        self.utterance_box.setCurrentIndex(index)

    @property
    def current_encoder_fpath(self):
        return self.encoder_box.itemData(self.encoder_box.currentIndex())
    
    @property
    def current_synthesizer_fpath(self):
        return self.synthesizer_box.itemData(self.synthesizer_box.currentIndex())
    
    @property
    def current_vocoder_fpath(self):
        return self.vocoder_box.itemData(self.vocoder_box.currentIndex())

    @property
    def current_extractor_fpath(self):
        return self.extractor_box.itemData(self.extractor_box.currentIndex())

    @property
    def current_convertor_fpath(self):
        return self.convertor_box.itemData(self.convertor_box.currentIndex())

    def populate_models(self, encoder_models_dir: Path, synthesizer_models_dir: Path, 
                        vocoder_models_dir: Path, extractor_models_dir: Path, convertor_models_dir: Path, vc_mode: bool):
        # Encoder
        encoder_fpaths = list(encoder_models_dir.glob("*.pt"))
        if len(encoder_fpaths) == 0:
            raise Exception("No encoder models found in %s" % encoder_models_dir)
        self.repopulate_box(self.encoder_box, [(f.stem, f) for f in encoder_fpaths])
        
        if vc_mode:
            # Extractor
            extractor_fpaths = list(extractor_models_dir.glob("*.pt"))
            if len(extractor_fpaths) == 0:
                self.log("No extractor models found in %s" % extractor_fpaths)
            self.repopulate_box(self.extractor_box, [(f.stem, f) for f in extractor_fpaths])
            
            # Convertor
            convertor_fpaths = list(convertor_models_dir.glob("*.pth"))
            if len(convertor_fpaths) == 0:
                self.log("No convertor models found in %s" % convertor_fpaths)
            self.repopulate_box(self.convertor_box, [(f.stem, f) for f in convertor_fpaths])
        else:
            # Synthesizer
            synthesizer_fpaths = list(synthesizer_models_dir.glob("**/*.pt"))
            if len(synthesizer_fpaths) == 0:
                raise Exception("No synthesizer models found in %s" % synthesizer_models_dir)
            self.repopulate_box(self.synthesizer_box, [(f.stem, f) for f in synthesizer_fpaths])

        # Vocoder
        vocoder_fpaths = list(vocoder_models_dir.glob("**/*.pt"))
        vocoder_items = [(f.stem, f) for f in vocoder_fpaths] + [("Griffin-Lim", None)]
        self.repopulate_box(self.vocoder_box, vocoder_items)

    @property
    def selected_utterance(self):
        return self.utterance_history.itemData(self.utterance_history.currentIndex())
        
    def register_utterance(self, utterance: Utterance, vc_mode):
        self.utterance_history.blockSignals(True)
        self.utterance_history.insertItem(0, utterance.name, utterance)
        self.utterance_history.setCurrentIndex(0)
        self.utterance_history.blockSignals(False)
        
        if len(self.utterance_history) > self.max_saved_utterances:
            self.utterance_history.removeItem(self.max_saved_utterances)

        self.play_button.setDisabled(False)
        if vc_mode:
            self.convert_button.setDisabled(False)
        else:
            self.generate_button.setDisabled(False)
            self.synthesize_button.setDisabled(False)

    def log(self, line, mode="newline"):
        if mode == "newline":
            self.logs.append(line)
            if len(self.logs) > self.max_log_lines:
                del self.logs[0]
        elif mode == "append":
            self.logs[-1] += line
        elif mode == "overwrite":
            self.logs[-1] = line
        log_text = '\n'.join(self.logs)
        
        self.log_window.setText(log_text)
        self.app.processEvents()

    def set_loading(self, value, maximum=1):
        self.loading_bar.setValue(value * 100)
        self.loading_bar.setMaximum(maximum * 100)
        self.loading_bar.setTextVisible(value != 0)
        self.app.processEvents()

    def populate_gen_options(self, seed, trim_silences):
        if seed is not None:
            self.random_seed_checkbox.setChecked(True)
            self.seed_textbox.setText(str(seed))
            self.seed_textbox.setEnabled(True)
        else:
            self.random_seed_checkbox.setChecked(False)
            self.seed_textbox.setText(str(0))
            self.seed_textbox.setEnabled(False)

        if not trim_silences:
            self.trim_silences_checkbox.setChecked(False)
            self.trim_silences_checkbox.setDisabled(True)

    def update_seed_textbox(self):
        if self.random_seed_checkbox.isChecked():
            self.seed_textbox.setEnabled(True)
        else:
            self.seed_textbox.setEnabled(False)

    def reset_interface(self, vc_mode):
        self.draw_embed(None, None, "current")
        self.draw_embed(None, None, "generated")
        self.draw_spec(None, "current")
        self.draw_spec(None, "generated")
        self.draw_umap_projections(set())
        self.set_loading(0)
        self.play_button.setDisabled(True)
        if vc_mode:
            self.convert_button.setDisabled(True)
        else:
            self.generate_button.setDisabled(True)
            self.synthesize_button.setDisabled(True)
        self.vocode_button.setDisabled(True)
        self.replay_wav_button.setDisabled(True)
        self.export_wav_button.setDisabled(True)
        [self.log("") for _ in range(self.max_log_lines)]

    #添加result_mfcc分析函数
    def plot_mfcc1(self, wav, sample_rate):

        signal = wav
        print(sample_rate, len(signal))
        # 读取前3.5s 的数据
        signal = signal[0:int(3.5 * sample_rate)]
        print(signal)

        # 预先处理
        pre_emphasis = 0.97
        emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        frame_size = 0.025
        frame_stride = 0.1
        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))

        pad_signal_length = num_frames * frame_step + frame_length
        z = numpy.zeros((pad_signal_length - signal_length))
        pad_signal = numpy.append(emphasized_signal, z)

        indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(
            numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

        frames = pad_signal[numpy.mat(indices).astype(numpy.int32, copy=False)]

        # 加上汉明窗
        frames *= numpy.hamming(frame_length)
        # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **

        # 傅立叶变换和功率谱
        NFFT = 512
        mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        # print(mag_frames.shape)
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

        low_freq_mel = 0
        # 将频率转换为Mel
        nfilt = 40
        high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))
        mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz

        bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))

        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = numpy.dot(pow_frames, fbank.T)
        filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * numpy.log10(filter_banks)  # dB

        # 所得到的倒谱系数2-13被保留，其余的被丢弃
        num_ceps = 12
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]
        (nframes, ncoeff) = mfcc.shape

        n = numpy.arange(ncoeff)
        cep_lifter = 22
        lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
        mfcc *= lift  # *

        # filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
        mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
        print(mfcc.shape)

        # 创建新的figure
        fig11 = plt.figure(figsize=(16,8))

        # 绘制1x2两行两列共四个图，编号从1开始
        ax = fig11.add_subplot(121)
        plt.plot(mfcc)

        ax = fig11.add_subplot(122)
        # 平均归一化MFCC
        mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
        plt.imshow(numpy.flipud(mfcc.T), cmap=plt.cm.jet, aspect=0.2,
                   extent=[0, mfcc.shape[0], 0, mfcc.shape[1]])  # 热力图
        #将figure保存为png并显示在新创建的子窗口上
        plt.savefig("fmcc_result.png")
        dialog_fault1 = QDialog()
        dialog_fault1.setWindowTitle("合成音频MFCC特征图及MFCC平均归一化热图")  # 设置窗口名
        pic = QPixmap("fmcc_result.png")
        label_pic = QLabel("show", dialog_fault1)
        label_pic.setPixmap(pic)
        label_pic.setGeometry(0,0,1500,800)
        dialog_fault1.exec_()



    def __init__(self, vc_mode):
        ## Initialize the application
        self.app = QApplication(sys.argv)



        super().__init__(None)
        self.setWindowTitle("中文语音克隆系统")
        self.setWindowIcon(QtGui.QIcon('toolbox\\assets\\mb.png'))
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)





        ## Main layouts
        # Root
        root_layout = QGridLayout()
        self.setLayout(root_layout)
        
        # Browser
        browser_layout = QGridLayout()
        root_layout.addLayout(browser_layout, 0, 0, 1, 8)
        
        # Generation
        gen_layout = QVBoxLayout()
        root_layout.addLayout(gen_layout, 0, 8)

        # Visualizations
        vis_layout = QVBoxLayout()
        root_layout.addLayout(vis_layout, 1, 0, 2, 8)

        # Output
        output_layout = QGridLayout()
        vis_layout.addLayout(output_layout, 0)

        # Projections
        self.projections_layout = QVBoxLayout()
        root_layout.addLayout(self.projections_layout, 1, 8, 2, 2)
        
        ## Projections
        # UMap
        fig, self.umap_ax = plt.subplots(figsize=(3, 3), facecolor="#F0F0F0")
        fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98)
        self.projections_layout.addWidget(FigureCanvas(fig))
        self.umap_hot = False
        self.clear_button = QPushButton("Clear")
        self.clear_button.setStyleSheet('QPushButton{border-image:url(toolbox/assets/2.png)}')
        self.projections_layout.addWidget(self.clear_button)


        ## Browser
        # Dataset, speaker and utterance selection
        i = 0

        source_groupbox = QGroupBox('Source(源音频)')
        source_layout = QGridLayout()
        source_groupbox.setLayout(source_layout)
        browser_layout.addWidget(source_groupbox, i, 0, 1, 5)

        self.dataset_box = QComboBox()
      #  source_layout.addWidget(QLabel("Dataset(数据集):"), i, 0)  #隐藏标签文字
        source_layout.addWidget(self.dataset_box, i, 1)
        self.random_dataset_button = QPushButton("Random")
        source_layout.addWidget(self.random_dataset_button, i, 2)

        self.random_dataset_button.hide()    #隐藏按钮
        self.dataset_box.hide()              #隐藏选项条

        i += 1
        self.speaker_box = QComboBox()
      #  source_layout.addWidget(QLabel("Speaker(说话者)"), i, 0)
        source_layout.addWidget(self.speaker_box, i, 1)
        self.random_speaker_button = QPushButton("Random")
        source_layout.addWidget(self.random_speaker_button, i, 2)

        self.random_speaker_button.hide()
        self.speaker_box.hide()

        i += 1
        self.utterance_box = QComboBox()
      #  source_layout.addWidget(QLabel("Utterance(音频):"), i, 0)
        source_layout.addWidget(self.utterance_box, i, 1)
        self.random_utterance_button = QPushButton("Random")
        source_layout.addWidget(self.random_utterance_button, i, 2)

        self.random_utterance_button.hide()
        self.utterance_box.hide()

        i += 1
        source_layout.addWidget(QLabel("<b>Use(使用):</b>"), i, 0)
        self.browser_load_button = QPushButton("")
        source_layout.addWidget(self.browser_load_button, i, 1, 1, 2)
        self.auto_next_checkbox = QCheckBox("Auto select next")
        self.auto_next_checkbox.setChecked(True)
        source_layout.addWidget(self.auto_next_checkbox, i + 1, 1)
        self.browser_browse_button = QPushButton("Browse(打开本地)")
        self.browser_browse_button.setStyleSheet('QPushButton{border-image:url(toolbox/assets/1.png)}')
        source_layout.addWidget(self.browser_browse_button, i, 3)
        self.record_button = QPushButton("Record(录音)")
        self.record_button.setStyleSheet('QPushButton{border-image:url(toolbox/assets/1.png)}')
        source_layout.addWidget(self.record_button, i+1, 3)
        
        i += 2
        # Utterance box
        browser_layout.addWidget(QLabel("<b>Current(当前):</b>"), i, 0)
        self.utterance_history = QComboBox()
        browser_layout.addWidget(self.utterance_history, i, 1)
        self.play_button = QPushButton("Play(播放)")
        self.play_button.setStyleSheet('QPushButton{border-image:url(toolbox/assets/1.png)}')
        browser_layout.addWidget(self.play_button, i, 2)
        self.stop_button = QPushButton("Stop(暂停)")
        self.stop_button.setStyleSheet('QPushButton{border-image:url(toolbox/assets/1.png)}')
        browser_layout.addWidget(self.stop_button, i, 3)
        if vc_mode:
            self.load_soruce_button = QPushButton("Select(选择为被转换的语音输入)")
            browser_layout.addWidget(self.load_soruce_button, i, 4)

        i += 1
        model_groupbox = QGroupBox('Models(模型选择)')
        model_layout = QHBoxLayout()
        model_groupbox.setLayout(model_layout)
        browser_layout.addWidget(model_groupbox, i, 0, 2, 5)

        # Model and audio output selection
        self.encoder_box = QComboBox()
        model_layout.addWidget(QLabel("Encoder:"))
        model_layout.addWidget(self.encoder_box)
        self.synthesizer_box = QComboBox()
        if vc_mode:
            self.extractor_box = QComboBox()
            model_layout.addWidget(QLabel("Extractor:"))
            model_layout.addWidget(self.extractor_box)
            self.convertor_box = QComboBox()
            model_layout.addWidget(QLabel("Convertor:"))
            model_layout.addWidget(self.convertor_box)
        else:
            model_layout.addWidget(QLabel("Synthesizer:"))
            model_layout.addWidget(self.synthesizer_box)
        self.vocoder_box = QComboBox()
        model_layout.addWidget(QLabel("Vocoder:"))
        model_layout.addWidget(self.vocoder_box)
    
        #Replay & Save Audio
        i = 0
        output_layout.addWidget(QLabel("<b>Toolbox Output:</b>"), i, 0)
        self.waves_cb = QComboBox()
        self.waves_cb_model = QStringListModel()
        self.waves_cb.setModel(self.waves_cb_model)
        self.waves_cb.setToolTip("Select one of the last generated waves in this section for replaying or exporting")
        output_layout.addWidget(self.waves_cb, i, 1)
        self.replay_wav_button = QPushButton("Replay(重播)")
        self.replay_wav_button.setStyleSheet('QPushButton{border-image:url(toolbox/assets/1.png)}')
        self.replay_wav_button.setToolTip("Replay last generated vocoder")
        output_layout.addWidget(self.replay_wav_button, i, 2)
        self.export_wav_button = QPushButton("Export(导出)")
        self.export_wav_button.setStyleSheet('QPushButton{border-image:url(toolbox/assets/1.png)}')
        self.export_wav_button.setToolTip("Save last generated vocoder audio in filesystem as a wav file")
        output_layout.addWidget(self.export_wav_button, i, 3)
        self.audio_out_devices_cb=QComboBox()
        i += 1
        output_layout.addWidget(QLabel("<b>Audio Output</b>"), i, 0)
        output_layout.addWidget(self.audio_out_devices_cb, i, 1)

        ## Embed & spectrograms
        vis_layout.addStretch()
        #添加标签控件，设置标签文字格式并且居中
        label1 = QLabel("source audio")
        label1.setStyleSheet("QLabel{color:red;font-size:20px;font-weight:bold;font-family:Roman times;}")
        label1.setAlignment(Qt.AlignCenter)
        vis_layout.addWidget(label1)      #addwidget:添加控件

        gridspec_kw = {"width_ratios": [1, 4]}
        fig, self.current_ax = plt.subplots(1, 2, figsize=(10, 2.25), facecolor="#F0F0F0",
                                            gridspec_kw=gridspec_kw)
        #self.current_ax[1].set_title("source audio", fontsize=50, color='red', fontstyle='italic', fontweight="heavy")
        fig.subplots_adjust(left=0, bottom=0.1, right=1, top=0.8)
        vis_layout.addWidget(FigureCanvas(fig))

        label2 = QLabel("target audio")
        label2.setStyleSheet("QLabel{color:red;font-size:20px;font-weight:bold;font-family:Roman times;}")
        label2.setAlignment(Qt.AlignCenter)
        vis_layout.addWidget(label2)

        fig, self.gen_ax = plt.subplots(1, 2, figsize=(10, 2.25), facecolor="#F0F0F0",
                                        gridspec_kw=gridspec_kw)
        #self.gen_ax[1].set_title("target audio", fontsize=50, color='red', fontstyle='italic', fontweight="heavy")
        fig.subplots_adjust(left=0, bottom=0.1, right=1, top=0.8)
        vis_layout.addWidget(FigureCanvas(fig))

        for ax in self.current_ax.tolist() + self.gen_ax.tolist():
            ax.set_facecolor("#F0F0F0")
            for side in ["top", "right", "bottom", "left"]:
                ax.spines[side].set_visible(False)





        ## Generation
        self.text_prompt = QPlainTextEdit(default_text)
        gen_layout.addWidget(self.text_prompt, stretch=1)
        
        self.generate_button = QPushButton("Synthesize and vocode(合成并播放)")
        self.generate_button.setStyleSheet('QPushButton{border-image:url(toolbox/assets/1.png)}')
        gen_layout.addWidget(self.generate_button)
        
        layout = QHBoxLayout()
        self.synthesize_button = QPushButton("Synthesize only(仅合成)")
        self.synthesize_button.setStyleSheet('QPushButton{border-image:url(toolbox/assets/1.png)}')
        layout.addWidget(self.synthesize_button)
        self.vocode_button = QPushButton("Vocode only(仅播放)")
        self.vocode_button.setStyleSheet('QPushButton{border-image:url(toolbox/assets/1.png)}')

        if vc_mode:
            layout = QHBoxLayout()
            self.convert_button = QPushButton("Extract and Convert")
            layout.addWidget(self.convert_button)
            gen_layout.addLayout(layout)
        else:
            self.generate_button = QPushButton("Synthesize and vocode")
            gen_layout.addWidget(self.generate_button)
            layout = QHBoxLayout()
            self.synthesize_button = QPushButton("Synthesize only")
            layout.addWidget(self.synthesize_button)

        self.vocode_button = QPushButton("Vocode only")
        layout.addWidget(self.vocode_button)
        gen_layout.addLayout(layout)


        layout_seed = QGridLayout()
        self.random_seed_checkbox = QCheckBox("Random seed(随机数种子):")
        self.random_seed_checkbox.setToolTip("When checked, makes the synthesizer and vocoder deterministic.")
        layout_seed.addWidget(self.random_seed_checkbox, 0, 0)
        self.seed_textbox = QLineEdit()
        self.seed_textbox.setMaximumWidth(80)
        layout_seed.addWidget(self.seed_textbox, 0, 1)
        self.trim_silences_checkbox = QCheckBox("Enhance vocoder output（语音增强）")
        self.trim_silences_checkbox.setToolTip("When checked, trims excess silence in vocoder output."
            " This feature requires `webrtcvad` to be installed.")
        layout_seed.addWidget(self.trim_silences_checkbox, 0, 2, 1, 2)
        self.style_slider = QSlider(Qt.Horizontal)
        self.style_slider.setTickInterval(1)
        self.style_slider.setFocusPolicy(Qt.NoFocus)
        self.style_slider.setSingleStep(1)
        self.style_slider.setRange(-1, 9)
        self.style_value_label = QLabel("-1")
        self.style_slider.setValue(-1)
        layout_seed.addWidget(QLabel("Style(风格):"), 1, 0)

        self.style_slider.valueChanged.connect(lambda s: self.style_value_label.setNum(s))
        layout_seed.addWidget(self.style_value_label, 1, 1)
        layout_seed.addWidget(self.style_slider, 1, 3)

        self.token_slider = QSlider(Qt.Horizontal)
        self.token_slider.setTickInterval(1)
        self.token_slider.setFocusPolicy(Qt.NoFocus)
        self.token_slider.setSingleStep(1)
        self.token_slider.setRange(3, 9)
        self.token_value_label = QLabel("4")
        self.token_slider.setValue(4)
        layout_seed.addWidget(QLabel("Accuracy(精度):"), 2, 0)

        self.token_slider.valueChanged.connect(lambda s: self.token_value_label.setNum(s))
        layout_seed.addWidget(self.token_value_label, 2, 1)
        layout_seed.addWidget(self.token_slider, 2, 3)

        self.length_slider = QSlider(Qt.Horizontal)
        self.length_slider.setTickInterval(1)
        self.length_slider.setFocusPolicy(Qt.NoFocus)
        self.length_slider.setSingleStep(1)
        self.length_slider.setRange(1, 10)
        self.length_value_label = QLabel("2")
        self.length_slider.setValue(2)
        layout_seed.addWidget(QLabel("MaxLength(最大句长):"), 3, 0)

        self.length_slider.valueChanged.connect(lambda s: self.length_value_label.setNum(s))
        layout_seed.addWidget(self.length_value_label, 3, 1)
        layout_seed.addWidget(self.length_slider, 3, 3)

        gen_layout.addLayout(layout_seed)

        self.loading_bar = QProgressBar()
        gen_layout.addWidget(self.loading_bar)
        
        self.log_window = QLabel()
        self.log_window.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        gen_layout.addWidget(self.log_window)
        self.logs = []
        gen_layout.addStretch()

        
        ## Set the size of the window and of the elements
        max_size = QDesktopWidget().availableGeometry(self).size() * 0.5
        self.resize(max_size)
        
        ## Finalize the display
        self.reset_interface(vc_mode)
        self.show()

        ##set the picture of background
        palette1 = QPalette()
        # palette1.setColor(self.backgroundRole(), QColor(192,253,123))   # 设置背景颜色
        palette1.setBrush(self.backgroundRole(), QBrush(QPixmap('toolbox\\assets\\picture1.jpg')))  # 设置背景图片
        self.setPalette(palette1)
        self.setAutoFillBackground(True)





    def start(self):
        self.app.exec_()
