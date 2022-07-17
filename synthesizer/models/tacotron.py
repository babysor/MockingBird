import torch
import torch.nn as nn
from .sublayer.global_style_token import GlobalStyleToken
from .sublayer.pre_net import PreNet
from .sublayer.cbhg import CBHG
from .sublayer.lsa import LSA
from .base import Base
from synthesizer.gst_hyperparameters import GSTHyperparameters as gst_hp
from synthesizer.hparams import hparams

class Encoder(nn.Module):
    def __init__(self, num_chars, embed_dims=512, encoder_dims=256, K=5, num_highways=4, dropout=0.5):
        """ Encoder for SV2TTS

        Args:
            num_chars (int): length of symbols
            embed_dims (int, optional): embedding dim for input texts. Defaults to 512.
            encoder_dims (int, optional): output dim for encoder. Defaults to 256.
            K (int, optional): _description_. Defaults to 5.
            num_highways (int, optional): _description_. Defaults to 4.
            dropout (float, optional): _description_. Defaults to 0.5.
        """             
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.pre_net = PreNet(embed_dims, fc1_dims=encoder_dims, fc2_dims=encoder_dims,
                              dropout=dropout)
        self.cbhg = CBHG(K=K, in_channels=encoder_dims, channels=encoder_dims,
                         proj_channels=[encoder_dims, encoder_dims],
                         num_highways=num_highways)

    def forward(self, x):
        """forward pass for encoder

        Args:
            x (2D tensor with size `[batch_size, text_num_chars]`): input texts list

        Returns:
            3D tensor with size `[batch_size, text_num_chars, encoder_dims]`
            
        """
        x = self.embedding(x) # return: [batch_size, text_num_chars, tts_embed_dims]
        x = self.pre_net(x) # return: [batch_size, text_num_chars, encoder_dims]
        x.transpose_(1, 2)  # return: [batch_size, encoder_dims, text_num_chars]
        return self.cbhg(x) # return: [batch_size, text_num_chars, encoder_dims]

class Decoder(nn.Module):
    # Class variable because its value doesn't change between classes
    # yet ought to be scoped by class because its a property of a Decoder
    max_r = 20
    def __init__(self, n_mels, input_dims, decoder_dims, lstm_dims,
                 dropout, speaker_embedding_size):
        super().__init__()
        self.register_buffer("r", torch.tensor(1, dtype=torch.int))
        self.n_mels = n_mels
        self.prenet = PreNet(n_mels, fc1_dims=decoder_dims * 2, fc2_dims=decoder_dims * 2,
                             dropout=dropout)
        self.attn_net = LSA(decoder_dims)
        if hparams.use_gst:
            speaker_embedding_size += gst_hp.E
        self.attn_rnn = nn.GRUCell(input_dims + decoder_dims * 2, decoder_dims)
        self.rnn_input = nn.Linear(input_dims  + decoder_dims, lstm_dims)
        self.res_rnn1 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.res_rnn2 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.mel_proj = nn.Linear(lstm_dims, n_mels * self.max_r, bias=False)
        self.stop_proj = nn.Linear(input_dims + lstm_dims, 1)

    def zoneout(self, prev, current, device, p=0.1):
        mask = torch.zeros(prev.size(),device=device).bernoulli_(p)
        return prev * mask + current * (1 - mask)

    def forward(self, encoder_seq, encoder_seq_proj, prenet_in,
                hidden_states, cell_states, context_vec, times, chars):
        """_summary_

        Args:
            encoder_seq (3D tensor `[batch_size, text_num_chars, project_dim(default to 512)]`): _description_
            encoder_seq_proj (3D tensor `[batch_size, text_num_chars, decoder_dims(default to 128)]`): _description_
            prenet_in (2D tensor `[batch_size, n_mels]`): _description_
            hidden_states (_type_): _description_
            cell_states (_type_): _description_
            context_vec (2D tensor `[batch_size, project_dim(default to 512)]`): _description_
            times (int): the number of times runned
            chars (2D tensor with size `[batch_size, text_num_chars]`): original texts list input

        """
        # Need this for reshaping mels
        batch_size = encoder_seq.size(0)
        device = encoder_seq.device
        # Unpack the hidden and cell states
        attn_hidden, rnn1_hidden, rnn2_hidden = hidden_states
        rnn1_cell, rnn2_cell = cell_states

        # PreNet for the Attention RNN
        prenet_out = self.prenet(prenet_in) # return: `[batch_size, decoder_dims * 2(256)]`

        # Compute the Attention RNN hidden state
        attn_rnn_in = torch.cat([context_vec, prenet_out], dim=-1) # `[batch_size, project_dim + decoder_dims * 2 (768)]`
        attn_hidden = self.attn_rnn(attn_rnn_in.squeeze(1), attn_hidden) #  `[batch_size, decoder_dims (128)]`

        # Compute the attention scores
        scores = self.attn_net(encoder_seq_proj, attn_hidden, times, chars)

        # Dot product to create the context vector
        context_vec = scores @ encoder_seq
        context_vec = context_vec.squeeze(1)

        # Concat Attention RNN output w. Context Vector & project
        x = torch.cat([context_vec, attn_hidden], dim=1) # `[batch_size, project_dim + decoder_dims (630)]`
        x = self.rnn_input(x) # `[batch_size, lstm_dims(1024)]`

        # Compute first Residual RNN, training with fixed zoneout rate 0.1
        rnn1_hidden_next, rnn1_cell = self.res_rnn1(x, (rnn1_hidden, rnn1_cell)) # `[batch_size, lstm_dims(1024)]`
        if self.training:
            rnn1_hidden = self.zoneout(rnn1_hidden, rnn1_hidden_next,device=device)
        else:
            rnn1_hidden = rnn1_hidden_next
        x = x + rnn1_hidden

        # Compute second Residual RNN
        rnn2_hidden_next, rnn2_cell = self.res_rnn2(x, (rnn2_hidden, rnn2_cell)) # `[batch_size, lstm_dims(1024)]`
        if self.training:
            rnn2_hidden = self.zoneout(rnn2_hidden, rnn2_hidden_next, device=device)
        else:
            rnn2_hidden = rnn2_hidden_next
        x = x + rnn2_hidden

        # Project Mels
        mels = self.mel_proj(x) # `[batch_size, 1600]`
        mels = mels.view(batch_size, self.n_mels, self.max_r)[:, :, :self.r] # `[batch_size, n_mels, r]`
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)
        cell_states = (rnn1_cell, rnn2_cell)

        # Stop token prediction
        s = torch.cat((x, context_vec), dim=1)
        s = self.stop_proj(s)
        stop_tokens = torch.sigmoid(s)

        return mels, scores, hidden_states, cell_states, context_vec, stop_tokens

class Tacotron(Base):
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, 
                 fft_bins, postnet_dims, encoder_K, lstm_dims, postnet_K, num_highways,
                 dropout, stop_threshold, speaker_embedding_size):
        super().__init__(stop_threshold)
        self.n_mels = n_mels
        self.lstm_dims = lstm_dims
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.speaker_embedding_size = speaker_embedding_size
        self.encoder = Encoder(num_chars, embed_dims, encoder_dims,
                               encoder_K, num_highways, dropout)
        self.project_dims = encoder_dims + speaker_embedding_size
        if hparams.use_gst: 
            self.project_dims += gst_hp.E
        self.encoder_proj = nn.Linear(self.project_dims, decoder_dims, bias=False)
        if hparams.use_gst: 
            self.gst = GlobalStyleToken(speaker_embedding_size)
        self.decoder = Decoder(n_mels, self.project_dims, decoder_dims, lstm_dims,
                               dropout, speaker_embedding_size)
        self.postnet = CBHG(postnet_K, n_mels, postnet_dims,
                            [postnet_dims, fft_bins], num_highways)
        self.post_proj = nn.Linear(postnet_dims, fft_bins, bias=False)

    @staticmethod
    def _concat_speaker_embedding(outputs, speaker_embeddings):
        speaker_embeddings_ = speaker_embeddings.expand(
            outputs.size(0), outputs.size(1), -1)
        outputs = torch.cat([outputs, speaker_embeddings_], dim=-1)
        return outputs

    @staticmethod
    def _add_speaker_embedding(x, speaker_embedding):
        """Add speaker embedding
            This concats the speaker embedding for each char in the encoder output
        Args:
            x (3D tensor with size `[batch_size, text_num_chars, encoder_dims]`): the encoder output
            speaker_embedding (2D tensor `[batch_size, speaker_embedding_size]`): the speaker embedding

        Returns:
            3D tensor with size `[batch_size, text_num_chars, encoder_dims+speaker_embedding_size]`
        """        
        # Save the dimensions as human-readable names
        batch_size = x.size()[0]
        text_num_chars = x.size()[1]

        # Start by making a copy of each speaker embedding to match the input text length
        # The output of this has size (batch_size, text_num_chars * speaker_embedding_size)
        speaker_embedding_size = speaker_embedding.size()[1]
        e = speaker_embedding.repeat_interleave(text_num_chars, dim=1)

        # Reshape it and transpose
        e = e.reshape(batch_size, speaker_embedding_size, text_num_chars)
        e = e.transpose(1, 2)

        # Concatenate the tiled speaker embedding with the encoder output
        x = torch.cat((x, e), 2)
        return x

    def forward(self, texts, mels, speaker_embedding, steps=2000, style_idx=0, min_stop_token=5):
        """Forward pass for Tacotron

        Args:
            texts (`[batch_size, text_num_chars]`): input texts list
            mels (`[batch_size, varied_mel_lengths, steps]`): mels for comparison (training only)
            speaker_embedding (`[batch_size, speaker_embedding_size(default to 256)]`): referring embedding.
            steps (int, optional): . Defaults to 2000.
            style_idx (int, optional): GST style selected. Defaults to 0.
            min_stop_token (int, optional): decoder min_stop_token. Defaults to 5.
        """
        device = texts.device  # use same device as parameters

        if self.training:
            self.step += 1
            batch_size, _, steps  = mels.size()
        else:
            batch_size, _  = texts.size()

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # SV2TTS: Run the encoder with the speaker embedding
        # The projection avoids unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(texts)
        
        encoder_seq = self._add_speaker_embedding(encoder_seq, speaker_embedding)

        if hparams.use_gst and self.gst is not None:
            if self.training:
                style_embed = self.gst(speaker_embedding, speaker_embedding) # for training, speaker embedding can represent both style inputs and referenced
                # style_embed = style_embed.expand_as(encoder_seq)
                # encoder_seq = torch.cat((encoder_seq, style_embed), 2)
            elif style_idx >= 0 and style_idx < 10:
                query = torch.zeros(1, 1, self.gst.stl.attention.num_units)
                if device.type == 'cuda':
                    query = query.cuda()
                gst_embed = torch.tanh(self.gst.stl.embed)
                key = gst_embed[style_idx].unsqueeze(0).expand(1, -1, -1)
                style_embed = self.gst.stl.attention(query, key)
            else:
                speaker_embedding_style = torch.zeros(speaker_embedding.size()[0], 1, self.speaker_embedding_size).to(device)
                style_embed = self.gst(speaker_embedding_style, speaker_embedding)
            encoder_seq = self._concat_speaker_embedding(encoder_seq, style_embed) # return: [batch_size, text_num_chars, project_dims]
        
        encoder_seq_proj = self.encoder_proj(encoder_seq) # return: [batch_size, text_num_chars, decoder_dims]

        # Need a couple of lists for outputs
        mel_outputs, attn_scores, stop_outputs = [], [], []

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.project_dims, device=device)

        # Run the decoder loop
        for t in range(0, steps, self.r):
            if self.training:
                prenet_in = mels[:, :, t -1] if t > 0 else go_frame
            else:
                prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
            mel_frames, scores, hidden_states, cell_states, context_vec, stop_tokens = \
                self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                             hidden_states, cell_states, context_vec, t, texts)
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)
            stop_outputs.extend([stop_tokens] * self.r)
            if not self.training and (stop_tokens * 10 > min_stop_token).all() and t > 10: break

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        # attn_scores = attn_scores.cpu().data.numpy()
        stop_outputs = torch.cat(stop_outputs, 1)

        if self.training:
            self.train()
            
        return mel_outputs, linear, attn_scores, stop_outputs

    def generate(self, x, speaker_embedding, steps=2000, style_idx=0, min_stop_token=5):
        self.eval()
        mel_outputs, linear, attn_scores, _ =  self.forward(x, None, speaker_embedding, steps, style_idx, min_stop_token)
        return mel_outputs, linear, attn_scores
