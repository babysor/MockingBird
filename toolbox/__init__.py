from toolbox.ui import UI
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder.wavernn import inference as rnn_vocoder
from vocoder.hifigan import inference as gan_vocoder
from vocoder.fregan import inference as fgan_vocoder
from pathlib import Path
from time import perf_counter as timer
from toolbox.utterance import Utterance
import numpy as np
import traceback
import sys
import torch
import re
from audioread.exceptions import NoBackendError
from specdeno.enhance_speach import enhance
import os
from synthesizer.hparams import hparams
import soundfile as sf

# 默认使用wavernn
vocoder = rnn_vocoder

# Use this directory structure for your datasets, or modify it to fit your needs
recognized_datasets = [
    "LibriSpeech/dev-clean",
    "LibriSpeech/dev-other",
    "LibriSpeech/test-clean",
    "LibriSpeech/test-other",
    "LibriSpeech/train-clean-100",
    "LibriSpeech/train-clean-360",
    "LibriSpeech/train-other-500",
    "LibriTTS/dev-clean",
    "LibriTTS/dev-other",
    "LibriTTS/test-clean",
    "LibriTTS/test-other",
    "LibriTTS/train-clean-100",
    "LibriTTS/train-clean-360",
    "LibriTTS/train-other-500",
    "LJSpeech-1.1",
    "VoxCeleb1/wav",
    "VoxCeleb1/test_wav",
    "VoxCeleb2/dev/aac",
    "VoxCeleb2/test/aac",
    "VCTK-Corpus/wav48",
    "aidatatang_200zh/corpus/dev",
    "aidatatang_200zh/corpus/test",
    "aishell3/test/wav",
    "magicdata/train",
]

#Maximum of generated wavs to keep on memory
MAX_WAVES = 15

class Toolbox:
    def __init__(self, datasets_root, enc_models_dir, syn_models_dir, voc_models_dir, extractor_models_dir, convertor_models_dir, seed, no_mp3_support, vc_mode):
        self.no_mp3_support = no_mp3_support
        self.vc_mode = vc_mode
        sys.excepthook = self.excepthook
        self.datasets_root = datasets_root
        self.utterances = set()
        self.current_generated = (None, None, None, None) # speaker_name, spec, breaks, wav
        
        self.synthesizer = None # type: Synthesizer

        # for ppg-based voice conversion
        self.extractor = None 
        self.convertor = None # ppg2mel

        self.current_wav = None
        self.waves_list = []
        self.waves_count = 0
        self.waves_namelist = []

        # Check for webrtcvad (enables removal of silences in vocoder output)
        try:
            import webrtcvad
            self.trim_silences = True
        except:
            self.trim_silences = False

        # Initialize the events and the interface
        self.ui = UI(vc_mode)
        self.style_idx = 0
        self.reset_ui(enc_models_dir, syn_models_dir, voc_models_dir, extractor_models_dir, convertor_models_dir, seed)
        self.setup_events()
        self.ui.start()

    def excepthook(self, exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)
        self.ui.log("Exception: %s" % exc_value)
        
    def setup_events(self):
        # Dataset, speaker and utterance selection
        self.ui.browser_load_button.clicked.connect(lambda: self.load_from_browser())
        random_func = lambda level: lambda: self.ui.populate_browser(self.datasets_root,
                                                                     recognized_datasets,
                                                                     level)
        self.ui.random_dataset_button.clicked.connect(random_func(0))
        self.ui.random_speaker_button.clicked.connect(random_func(1))
        self.ui.random_utterance_button.clicked.connect(random_func(2))
        self.ui.dataset_box.currentIndexChanged.connect(random_func(1))
        self.ui.speaker_box.currentIndexChanged.connect(random_func(2))
        
        # Model selection
        self.ui.encoder_box.currentIndexChanged.connect(self.init_encoder)
        def func(): 
            self.synthesizer = None
        if self.vc_mode:
            self.ui.extractor_box.currentIndexChanged.connect(self.init_extractor)
        else:
            self.ui.synthesizer_box.currentIndexChanged.connect(func)

        self.ui.vocoder_box.currentIndexChanged.connect(self.init_vocoder)
        
        # Utterance selection
        func = lambda: self.load_from_browser(self.ui.browse_file())
        self.ui.browser_browse_button.clicked.connect(func)
        func = lambda: self.ui.draw_utterance(self.ui.selected_utterance, "current")
        self.ui.utterance_history.currentIndexChanged.connect(func)
        func = lambda: self.ui.play(self.ui.selected_utterance.wav, Synthesizer.sample_rate)
        self.ui.play_button.clicked.connect(func)
        self.ui.stop_button.clicked.connect(self.ui.stop)
        self.ui.record_button.clicked.connect(self.record)

        #添加source_mfcc分析槽
        func = lambda: self.ui.plot_mfcc(self.ui.selected_utterance.wav, Synthesizer.sample_rate)
        self.ui.play_button.clicked.connect(func)

        # Source Utterance selection
        if self.vc_mode:
            func = lambda: self.load_soruce_button(self.ui.selected_utterance)
            self.ui.load_soruce_button.clicked.connect(func)
        #Audio
        self.ui.setup_audio_devices(Synthesizer.sample_rate)

        #Wav playback & save
        func = lambda: self.replay_last_wav()
        self.ui.replay_wav_button.clicked.connect(func)
        func = lambda: self.export_current_wave()
        self.ui.export_wav_button.clicked.connect(func)
        self.ui.waves_cb.currentIndexChanged.connect(self.set_current_wav)



        # Generation
        self.ui.vocode_button.clicked.connect(self.vocode)
        self.ui.random_seed_checkbox.clicked.connect(self.update_seed_textbox)

        # 添加result_mfcc分析槽,该槽要在语音合成之后
        func = lambda: self.ui.plot_mfcc1(self.current_wav, Synthesizer.sample_rate)
        self.ui.generate_button.clicked.connect(func)
        if self.vc_mode:
            func = lambda: self.convert() or self.vocode()
            self.ui.convert_button.clicked.connect(func)
        else:
            func = lambda: self.synthesize() or self.vocode()
            self.ui.generate_button.clicked.connect(func)
            self.ui.synthesize_button.clicked.connect(self.synthesize)

        # UMAP legend
        self.ui.clear_button.clicked.connect(self.clear_utterances)

    def set_current_wav(self, index):
        self.current_wav = self.waves_list[index]

    def export_current_wave(self):
        self.ui.save_audio_file(self.current_wav, Synthesizer.sample_rate)

    def replay_last_wav(self):
        self.ui.play(self.current_wav, Synthesizer.sample_rate)

    def reset_ui(self, encoder_models_dir, synthesizer_models_dir, vocoder_models_dir, extractor_models_dir, convertor_models_dir, seed):
        self.ui.populate_browser(self.datasets_root, recognized_datasets, 0, True)
        self.ui.populate_models(encoder_models_dir, synthesizer_models_dir, vocoder_models_dir, extractor_models_dir, convertor_models_dir, self.vc_mode)
        self.ui.populate_gen_options(seed, self.trim_silences)
        
    def load_from_browser(self, fpath=None):
        if fpath is None:
            fpath = Path(self.datasets_root,
                         self.ui.current_dataset_name,
                         self.ui.current_speaker_name,
                         self.ui.current_utterance_name)
            name = str(fpath.relative_to(self.datasets_root))
            speaker_name = self.ui.current_dataset_name + '_' + self.ui.current_speaker_name
            
            # Select the next utterance
            if self.ui.auto_next_checkbox.isChecked():
                self.ui.browser_select_next()
        elif fpath == "":
            return 
        else:
            name = fpath.name
            speaker_name = fpath.parent.name

        if fpath.suffix.lower() == ".mp3" and self.no_mp3_support:
                self.ui.log("Error: No mp3 file argument was passed but an mp3 file was used")
                return

        # Get the wav from the disk. We take the wav with the vocoder/synthesizer format for
        # playback, so as to have a fair comparison with the generated audio
        #wav = Synthesizer.load_preprocess_wav(fpath)
        wav = enhance(fpath)

        self.ui.log("Loaded %s" % name)

        self.add_real_utterance(wav, name, speaker_name)
    
    def load_soruce_button(self, utterance: Utterance):
        self.selected_source_utterance = utterance

    def record(self):
        wav = self.ui.record_one(encoder.sampling_rate, 5)
        sf.write('output1.wav', wav, hparams.sample_rate)  # 先将变量wav写为文件的形式
        wav = enhance('output1.wav')
        os.remove("./output1.wav")
        if wav is None:
            return 
        self.ui.play(wav, encoder.sampling_rate)

        speaker_name = "user01"
        name = speaker_name + "_rec_%05d" % np.random.randint(100000)
        self.add_real_utterance(wav, name, speaker_name)
        
    def add_real_utterance(self, wav, name, speaker_name):
        # Compute the mel spectrogram
        spec = Synthesizer.make_spectrogram(wav)
        self.ui.draw_spec(spec, "current")

        # Compute the embedding
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        # Add the utterance
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, False)
        self.utterances.add(utterance)
        self.ui.register_utterance(utterance, self.vc_mode)

        # Plot it
        self.ui.draw_embed(embed, name, "current")
        self.ui.draw_umap_projections(self.utterances)
        
    def clear_utterances(self):
        self.utterances.clear()
        self.ui.draw_umap_projections(self.utterances)
        
    def synthesize(self):
        self.ui.log("Generating the mel spectrogram...")
        self.ui.set_loading(1)
        
        # Update the synthesizer random seed
        if self.ui.random_seed_checkbox.isChecked():
            seed = int(self.ui.seed_textbox.text())
            self.ui.populate_gen_options(seed, self.trim_silences)
        else:
            seed = None

        if seed is not None:
            torch.manual_seed(seed)

        # Synthesize the spectrogram
        if self.synthesizer is None or seed is not None:
            self.init_synthesizer()

        texts = self.ui.text_prompt.toPlainText().split("\n")
        punctuation = '！，。、,' # punctuate and split/clean text
        processed_texts = []
        for text in texts:
          for processed_text in re.sub(r'[{}]+'.format(punctuation), '\n', text).split('\n'):
            if processed_text:
                processed_texts.append(processed_text.strip())
        texts = processed_texts
        embed = self.ui.selected_utterance.embed
        embeds = [embed] * len(texts)
        min_token = int(self.ui.token_slider.value())
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds, style_idx=int(self.ui.style_slider.value()), min_stop_token=min_token, steps=int(self.ui.length_slider.value())*200)
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)
        
        self.ui.draw_spec(spec, "generated")
        self.current_generated = (self.ui.selected_utterance.speaker_name, spec, breaks, None)
        self.ui.set_loading(0)

    def vocode(self):
        speaker_name, spec, breaks, _ = self.current_generated
        assert spec is not None

        # Initialize the vocoder model and make it determinstic, if user provides a seed
        if self.ui.random_seed_checkbox.isChecked():
            seed = int(self.ui.seed_textbox.text())
            self.ui.populate_gen_options(seed, self.trim_silences)
        else:
            seed = None

        if seed is not None:
            torch.manual_seed(seed)

        # Synthesize the waveform
        if not vocoder.is_loaded() or seed is not None:
            self.init_vocoder()

        def vocoder_progress(i, seq_len, b_size, gen_rate):
            real_time_factor = (gen_rate / Synthesizer.sample_rate) * 1000
            line = "Waveform generation: %d/%d (batch size: %d, rate: %.1fkHz - %.2fx real time)" \
                   % (i * b_size, seq_len * b_size, b_size, gen_rate, real_time_factor)
            self.ui.log(line, "overwrite")
            self.ui.set_loading(i, seq_len)
        if self.ui.current_vocoder_fpath is not None:
            self.ui.log("")
            wav, sample_rate = vocoder.infer_waveform(spec, progress_callback=vocoder_progress)
        else:
            self.ui.log("Waveform generation with Griffin-Lim... ")
            wav = Synthesizer.griffin_lim(spec)
        self.ui.set_loading(0)
        self.ui.log(" Done!", "append")
        
        # Add breaks
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        # Trim excessive silences
        if self.ui.trim_silences_checkbox.isChecked():
            #wav = encoder.preprocess_wav(wav)
            sf.write('output.wav', wav, hparams.sample_rate)      #先将变量wav写为文件的形式
            wav = enhance('output.wav')
            os.remove("./output.wav")

        # Play it
        wav = wav / np.abs(wav).max() * 0.97
        self.ui.play(wav, sample_rate)

        # Name it (history displayed in combobox)
        # TODO better naming for the combobox items?
        wav_name = str(self.waves_count + 1)

        #Update waves combobox
        self.waves_count += 1
        if self.waves_count > MAX_WAVES:
          self.waves_list.pop()
          self.waves_namelist.pop()
        self.waves_list.insert(0, wav)
        self.waves_namelist.insert(0, wav_name)

        self.ui.waves_cb.disconnect()
        self.ui.waves_cb_model.setStringList(self.waves_namelist)
        self.ui.waves_cb.setCurrentIndex(0)
        self.ui.waves_cb.currentIndexChanged.connect(self.set_current_wav)

        # Update current wav
        self.set_current_wav(0)
        
        #Enable replay and save buttons:
        self.ui.replay_wav_button.setDisabled(False)
        self.ui.export_wav_button.setDisabled(False)

        # Compute the embedding
        # TODO: this is problematic with different sampling rates, gotta fix it
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
        
        # Add the utterance
        name = speaker_name + "_gen_%05d" % np.random.randint(100000)
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, True)
        self.utterances.add(utterance)
        
        # Plot it
        self.ui.draw_embed(embed, name, "generated")
        self.ui.draw_umap_projections(self.utterances)
        
    def convert(self):
        self.ui.log("Extract PPG and Converting...")
        self.ui.set_loading(1)
        
        # Init
        if self.convertor is None:
            self.init_convertor()
        if self.extractor is None:
            self.init_extractor()
        
        src_wav = self.selected_source_utterance.wav

        # Compute the ppg
        if not self.extractor is None:
            ppg = self.extractor.extract_from_wav(src_wav)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ref_wav = self.ui.selected_utterance.wav
        # Import necessary dependency of Voice Conversion
        from utils.f0_utils import compute_f0, f02lf0, compute_mean_std, get_converted_lf0uv   
        ref_lf0_mean, ref_lf0_std = compute_mean_std(f02lf0(compute_f0(ref_wav)))
        lf0_uv = get_converted_lf0uv(src_wav, ref_lf0_mean, ref_lf0_std, convert=True)
        min_len = min(ppg.shape[1], len(lf0_uv))
        ppg = ppg[:, :min_len]
        lf0_uv = lf0_uv[:min_len]
        _, mel_pred, att_ws = self.convertor.inference(
            ppg,
            logf0_uv=torch.from_numpy(lf0_uv).unsqueeze(0).float().to(device),
            spembs=torch.from_numpy(self.ui.selected_utterance.embed).unsqueeze(0).to(device),
        )
        mel_pred= mel_pred.transpose(0, 1)
        breaks = [mel_pred.shape[1]]
        mel_pred= mel_pred.detach().cpu().numpy()
        self.ui.draw_spec(mel_pred, "generated")
        self.current_generated = (self.ui.selected_utterance.speaker_name, mel_pred, breaks, None)
        self.ui.set_loading(0)

    def init_extractor(self):
        if self.ui.current_extractor_fpath is None:
            return
        model_fpath = self.ui.current_extractor_fpath
        self.ui.log("Loading the extractor %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        import ppg_extractor as extractor
        self.extractor = extractor.load_model(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    def init_convertor(self):
        if self.ui.current_convertor_fpath is None:
            return
        model_fpath = self.ui.current_convertor_fpath
        # search a config file
        model_config_fpaths = list(model_fpath.parent.rglob("*.yaml"))
        if self.ui.current_convertor_fpath is None:
            return
        model_config_fpath = model_config_fpaths[0]
        self.ui.log("Loading the convertor %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        import ppg2mel as convertor
        self.convertor = convertor.load_model(model_config_fpath, model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)
        
    def init_encoder(self):
        model_fpath = self.ui.current_encoder_fpath
        
        self.ui.log("Loading the encoder %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        encoder.load_model(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    def init_synthesizer(self):
        model_fpath = self.ui.current_synthesizer_fpath

        self.ui.log("Loading the synthesizer %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        self.synthesizer = Synthesizer(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)
           
    def init_vocoder(self):

        global vocoder
        model_fpath = self.ui.current_vocoder_fpath
        # Case of Griffin-lim
        if model_fpath is None:
            return 

        # Select vocoder based on model name
        if model_fpath.name[0] == "g":
            vocoder = gan_vocoder
            self.ui.log("set hifigan as vocoder")
        elif model_fpath.name[0] == "m":
            vocoder = fgan_vocoder
            self.ui.log("set fregan as vocoder")
        # Sekect vocoder based on model name
        model_config_fpath = None
        if model_fpath.name[0] == "g":
            vocoder = gan_vocoder
            self.ui.log("set hifigan as vocoder")
            # search a config file
            model_config_fpaths = list(model_fpath.parent.rglob("*.json"))
            if self.vc_mode and self.ui.current_extractor_fpath is None:
                return
            if len(model_config_fpaths) > 0:
                model_config_fpath = model_config_fpaths[0]
        else:
            vocoder = rnn_vocoder
            self.ui.log("set wavernn as vocoder")
    
        self.ui.log("Loading the vocoder %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        vocoder.load_model(model_fpath, model_config_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    def update_seed_textbox(self):
       self.ui.update_seed_textbox()


