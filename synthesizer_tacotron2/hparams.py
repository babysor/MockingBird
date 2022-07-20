import ast
import pprint
from tensorflow.contrib.training import HParams



hparams = HParams(
        cleaners="basic_cleaners",
        tacotron_gpu_start_idx=0,  # idx of the first GPU to be used for Tacotron training.
        tacotron_num_gpus=1,  # Determines the number of gpus in use for Tacotron training.
        split_on_cpu=True,

        ### Signal Processing (used in both synthesizer and vocoder)
        sample_rate = 16000,
        n_fft = 800,
        num_mels = 80,
        hop_size = 200,                             # Tacotron uses 12.5 ms frame shift (set to sample_rate * 0.0125)
        win_size = 800,                             # Tacotron uses 50 ms frame length (set to sample_rate * 0.050)
        fmin = 55,
        min_level_db = -100,
        ref_level_db = 20,
        max_abs_value = 4.,                         # Gradient explodes if too big, premature convergence if too small.
        preemphasis = 0.97,                         # Filter coefficient to use if preemphasize is True
        preemphasize = True,
        frame_shift_ms=None,
        normalize_for_wavenet=True,
        # whether to rescale to [0, 1] for wavenet. (better audio quality)
        clip_for_wavenet=True,



        ### Tacotron Text-to-Speech (TTS)
        tts_embed_dims = 512,                       # Embedding dimension for the graphemes/phoneme inputs
        tts_encoder_dims = 256,
        tts_decoder_dims = 128,
        tts_postnet_dims = 512,
        tts_encoder_K = 5,
        tts_lstm_dims = 1024,
        tts_postnet_K = 5,
        tts_num_highways = 4,
        tts_dropout = 0.5,
        tts_cleaner_names = ["basic_cleaners"],
        tts_stop_threshold = -3.4,                  # Value below which audio generation ends.
                                                    # For example, for a range of [-4, 4], this
                                                    # will terminate the sequence at the first
                                                    # frame that has all values < -3.4

        ### Tacotron Training
        tts_schedule = [(2,  1e-3,  20_000,  12),   # Progressive training schedule
                        (2,  5e-4,  40_000,  12),   # (r, lr, step, batch_size)
                        (2,  2e-4,  80_000,  12),   #
                        (2,  1e-4, 160_000,  12),   # r = reduction factor (# of mel frames
                        (2,  3e-5, 320_000,  12),   #     synthesized for each decoder iteration)
                        (2,  1e-5, 640_000,  12)],  # lr = learning rate

        tts_clip_grad_norm = 1.0,                   # clips the gradient norm to prevent explosion - set to None if not needed
        tts_eval_interval = 500,                    # Number of steps between model evaluation (sample generation)
                                                    # Set to -1 to generate after completing epoch, or 0 to disable

        tts_eval_num_samples = 1,                   # Makes this number of samples

        ### Data Preprocessing
        max_mel_frames = 900,
        rescale = True,
        rescaling_max = 0.9,
        synthesis_batch_size = 16,                  # For vocoder preprocessing and inference.

        ### Mel Visualization and Griffin-Lim
        signal_normalization = True,
        power = 1.5,
        griffin_lim_iters = 60,

        ### Audio processing options
        fmax = 7600,                                # Should not exceed (sample_rate // 2)
        allow_clipping_in_normalization = True,     # Used when signal_normalization = True
        clip_mels_length = True,                    # If true, discards samples exceeding max_mel_frames
        use_lws = False,                            # "Fast spectrogram phase recovery using local weighted sums"
        symmetric_mels = True,                      # Sets mel range to [-max_abs_value, max_abs_value] if True,
                                                    #               and [0, max_abs_value] if False
        trim_silence = True,                        # Use with sample_rate of 16000 for best results
        silence_threshold=2,
        trim_fft_size=512,
        trim_hop_size=128,
        trim_top_db=23,

        ### SV2TTS
        speaker_embedding_size = 256,               # Dimension for the speaker embedding
        silence_min_duration_split = 0.4,           # Duration in seconds of a silence for an utterance to be split
        utterance_min_duration = 1.6,               # Duration in seconds below which utterances are discarded

        # Tacotron
        outputs_per_step=2,  # Was 1
        # number of frames to generate at each decoding step (increase to speed up computation and
        # allows for higher batch size, decreases G&L audio quality)
        stop_at_any=True,
        # Determines whether the decoder should stop when predicting <stop> to any frame or to all of
        # them (True works pretty well)

        embedding_dim=512,  # dimension of embedding space (these are NOT the speaker embeddings)

        # Encoder parameters
        enc_conv_num_layers=3,  # number of encoder convolutional layers
        enc_conv_kernel_size=(5,),  # size of encoder convolution filters for each layer
        enc_conv_channels=512,  # number of encoder convolutions filters for each layer
        encoder_lstm_units=256,  # number of lstm units for each direction (forward and backward)

        # Attention mechanism
        smoothing=False,  # Whether to smooth the attention normalization function
        attention_dim=128,  # dimension of attention space
        attention_filters=32,  # number of attention convolution filters
        attention_kernel=(31,),  # kernel size of attention convolution
        cumulative_weights=True,
        # Whether to cumulate (sum) all previous attention weights or simply feed previous weights (
        # Recommended: True)

        # Decoder
        prenet_layers=[256, 256],  # number of layers and number of units of prenet
        decoder_layers=2,  # number of decoder lstm layers
        decoder_lstm_units=1024,  # number of decoder lstm units on each layer
        max_iters=2000,
        # Max decoder steps during inference (Just for safety from infinite loop cases)

        # Residual postnet
        postnet_num_layers=5,  # number of postnet convolutional layers
        postnet_kernel_size=(5,),  # size of postnet convolution filters for each layer
        postnet_channels=512,  # number of postnet convolution filters for each layer

        # CBHG mel->linear postnet
        cbhg_kernels=8,
        # All kernel sizes from 1 to cbhg_kernels will be used in the convolution bank of CBHG to act
        #  as "K-grams"
        cbhg_conv_channels=128,  # Channels of the convolution bank
        cbhg_pool_size=2,  # pooling size of the CBHG
        cbhg_projection=256,
        # projection channels of the CBHG (1st projection, 2nd is automatically set to num_mels)
        cbhg_projection_kernel_size=3,  # kernel_size of the CBHG projections
        cbhg_highwaynet_layers=4,  # Number of HighwayNet layers
        cbhg_highway_units=128,  # Number of units used in HighwayNet fully connected layers
        cbhg_rnn_units=128,
        # Number of GRU units used in bidirectional RNN of CBHG block. CBHG output is 2x rnn_units in
        # shape

        # Loss params
        mask_encoder=True,
        # whether to mask encoder padding while computing attention. Set to True for better prosody
        # but slower convergence.
        mask_decoder=False,
        # Whether to use loss mask for padded sequences (if False, <stop_token> loss function will not
        #  be weighted, else recommended pos_weight = 20)
        cross_entropy_pos_weight=20,
        # Use class weights to reduce the stop token classes imbalance (by adding more penalty on
        # False Negatives (FN)) (1 = disabled)
        predict_linear=False,
        # Whether to add a post-processing network to the Tacotron to predict linear spectrograms (
        # True mode Not tested!!)
        ###########################################################################################################################################

        # Tacotron Training
        # Reproduction seeds
        tacotron_random_seed=5339,
        # Determines initial graph and operations (i.e: model) random state for reproducibility
        tacotron_data_random_state=1234,  # random state for train test split repeatability

        # performance parameters
        tacotron_swap_with_cpu=False,
        # Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause
        # major slowdowns! Only use when critical!)

        # train/test split ratios, mini-batches sizes
        tacotron_batch_size=36,  # number of training samples on each training steps (was 32)
        # Tacotron Batch synthesis supports ~16x the training batch size (no gradients during
        # testing).
        # Training Tacotron with unmasked paddings makes it aware of them, which makes synthesis times
        #  different from training. We thus recommend masking the encoder.
        tacotron_synthesis_batch_size=128,
        # DO NOT MAKE THIS BIGGER THAN 1 IF YOU DIDN"T TRAIN TACOTRON WITH "mask_encoder=True"!!
        tacotron_test_size=0.05,
        # % of data to keep as test data, if None, tacotron_test_batches must be not None. (5% is
        # enough to have a good idea about overfit)
        tacotron_test_batches=None,  # number of test batches.

        # Learning rate schedule
        tacotron_decay_learning_rate=True,
        # boolean, determines if the learning rate will follow an exponential decay
        tacotron_start_decay=50000,  # Step at which learning decay starts
        tacotron_decay_steps=50000,  # Determines the learning rate decay slope (UNDER TEST)
        tacotron_decay_rate=0.5,  # learning rate decay rate (UNDER TEST)
        tacotron_initial_learning_rate=1e-3,  # starting learning rate
        tacotron_final_learning_rate=1e-5,  # minimal learning rate

        # Optimization parameters
        tacotron_adam_beta1=0.9,  # AdamOptimizer beta1 parameter
        tacotron_adam_beta2=0.999,  # AdamOptimizer beta2 parameter
        tacotron_adam_epsilon=1e-6,  # AdamOptimizer Epsilon parameter

        # Regularization parameters
        tacotron_reg_weight=1e-7,  # regularization weight (for L2 regularization)
        tacotron_scale_regularization=False,
        # Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is
        #  high and biasing the model)
        tacotron_zoneout_rate=0.1,  # zoneout rate for all LSTM cells in the network
        tacotron_dropout_rate=0.5,  # dropout rate for all convolutional layers + prenet
        tacotron_clip_gradients=True,  # whether to clip gradients

        # Evaluation parameters
        natural_eval=False,
        # Whether to use 100% natural eval (to evaluate Curriculum Learning performance) or with same
        #  teacher-forcing ratio as in training (just for overfit)

        # Decoder RNN learning can take be done in one of two ways:
        #	Teacher Forcing: vanilla teacher forcing (usually with ratio = 1). mode="constant"
        #	Curriculum Learning Scheme: From Teacher-Forcing to sampling from previous outputs is
        # function of global step. (teacher forcing ratio decay) mode="scheduled"
        # The second approach is inspired by:
        # Bengio et al. 2015: Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.
        # Can be found under: https://arxiv.org/pdf/1506.03099.pdf
        tacotron_teacher_forcing_mode="constant",
        # Can be ("constant" or "scheduled"). "scheduled" mode applies a cosine teacher forcing ratio
        # decay. (Preference: scheduled)
        tacotron_teacher_forcing_ratio=1.,
        # Value from [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder
        # inputs, Only relevant if mode="constant"
        tacotron_teacher_forcing_init_ratio=1.,
        # initial teacher forcing ratio. Relevant if mode="scheduled"
        tacotron_teacher_forcing_final_ratio=0.,
        # final teacher forcing ratio. Relevant if mode="scheduled"
        tacotron_teacher_forcing_start_decay=10000,
        # starting point of teacher forcing ratio decay. Relevant if mode="scheduled"
        tacotron_teacher_forcing_decay_steps=280000,
        # Determines the teacher forcing ratio decay slope. Relevant if mode="scheduled"
        tacotron_teacher_forcing_decay_alpha=0.,
        # teacher forcing ratio decay rate. Relevant if mode="scheduled"
        ###########################################################################################################################################

        # Tacotron-2 integration parameters
        train_with_GTA=False,
        # Whether to use GTA mels to train WaveNet instead of ground truth mels.
        ###########################################################################################################################################

        # Eval sentences (if no eval text file was specified during synthesis, these sentences are
        # used for eval)
        sentences=[
            # From July 8, 2017 New York Times:
            "Scientists at the CERN laboratory say they have discovered a new particle.",
            "There\"s a way to measure the acute emotional intelligence that has never gone out of "
            "style.",
            "President Trump met with other leaders at the Group of 20 conference.",
            "The Senate\"s bill to repeal and replace the Affordable Care Act is now imperiled.",
            # From Google"s Tacotron example page:
            "Generative adversarial network or variational auto-encoder.",
            "Basilar membrane and otolaryngology are not auto-correlations.",
            "He has read the whole thing.",
            "He reads books.",
            "He thought it was time to present the present.",
            "Thisss isrealy awhsome.",
            "Punctuation sensitivity, is working.",
            "Punctuation sensitivity is working.",
            "Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?",
            "She sells sea-shells on the sea-shore. The shells she sells are sea-shells I'm sure.",
            "Tajima Airport serves Toyooka.",
            # From The web (random long utterance)
            "Sequence to sequence models have enjoyed great success in a variety of tasks such as machine translation, speech recognition, and text summarization.\
            This project covers a sequence to sequence model trained to predict a speech representation from an input sequence of characters. We show that\
            the adopted architecture is able to perform this task with wild success.",
            "Thank you so much for your support!",
        ],
        )

def hparams_debug_string():
    values = hparams.values()
    hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
    return "Hyperparameters:\n" + "\n".join(hp)