import torch
import torch.nn as nn
import torch.nn.functional as F


class MOLAttention(nn.Module):
    """ Discretized Mixture of Logistic (MOL) attention.
    C.f. Section 5 of "MelNet: A Generative Model for Audio in the Frequency Domain" and 
        GMMv2b model in "Location-relative attention mechanisms for robust long-form speech synthesis".
    """
    def __init__(
        self,
        query_dim,
        r=1,
        M=5,
    ):
        """
        Args:
            query_dim: attention_rnn_dim.
            M: number of mixtures.
        """
        super().__init__()
        if r < 1:
            self.r = float(r)
        else:
            self.r = int(r)
        self.M = M
        self.score_mask_value = 0.0 # -float("inf")
        self.eps = 1e-5
        # Position arrary for encoder time steps
        self.J = None
        # Query layer: [w, sigma,]
        self.query_layer = torch.nn.Sequential(
            nn.Linear(query_dim, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 3*M, bias=True)
        )
        self.mu_prev = None
        self.initialize_bias()

    def initialize_bias(self):
        """Initialize sigma and Delta."""
        # sigma
        torch.nn.init.constant_(self.query_layer[2].bias[self.M:2*self.M], 1.0)
        # Delta: softplus(1.8545) = 2.0; softplus(3.9815) = 4.0; softplus(0.5413) = 1.0
        # softplus(-0.432) = 0.5003
        if self.r == 2:
            torch.nn.init.constant_(self.query_layer[2].bias[2*self.M:3*self.M], 1.8545)
        elif self.r == 4:
            torch.nn.init.constant_(self.query_layer[2].bias[2*self.M:3*self.M], 3.9815)
        elif self.r == 1:
            torch.nn.init.constant_(self.query_layer[2].bias[2*self.M:3*self.M], 0.5413)
        else:
            torch.nn.init.constant_(self.query_layer[2].bias[2*self.M:3*self.M], -0.432)

    
    def init_states(self, memory):
        """Initialize mu_prev and J.
            This function should be called by the decoder before decoding one batch.
        Args:
            memory: (B, T, D_enc) encoder output.
        """
        B, T_enc, _ = memory.size()
        device = memory.device
        self.J = torch.arange(0, T_enc + 2.0).to(device) + 0.5  # NOTE: for discretize usage
        # self.J = memory.new_tensor(np.arange(T_enc), dtype=torch.float)
        self.mu_prev = torch.zeros(B, self.M).to(device)

    def forward(self, att_rnn_h, memory, memory_pitch=None, mask=None):
        """
        att_rnn_h: attetion rnn hidden state.
        memory: encoder outputs (B, T_enc, D).
        mask: binary mask for padded data (B, T_enc).
        """
        # [B, 3M]
        mixture_params = self.query_layer(att_rnn_h)
        
        # [B, M]
        w_hat = mixture_params[:, :self.M]
        sigma_hat = mixture_params[:, self.M:2*self.M]
        Delta_hat = mixture_params[:, 2*self.M:3*self.M]
        
        # print("w_hat: ", w_hat)
        # print("sigma_hat: ", sigma_hat)
        # print("Delta_hat: ", Delta_hat)

        # Dropout to de-correlate attention heads
        w_hat = F.dropout(w_hat, p=0.5, training=self.training) # NOTE(sx): needed?
        
        # Mixture parameters
        w = torch.softmax(w_hat, dim=-1) + self.eps
        sigma = F.softplus(sigma_hat) + self.eps
        Delta = F.softplus(Delta_hat)
        mu_cur = self.mu_prev + Delta
        # print("w:", w)
        j = self.J[:memory.size(1) + 1]

        # Attention weights
        # CDF of logistic distribution
        phi_t = w.unsqueeze(-1) * (1 / (1 + torch.sigmoid(
            (mu_cur.unsqueeze(-1) - j) / sigma.unsqueeze(-1))))
        # print("phi_t:", phi_t)
        
        # Discretize attention weights
        # (B, T_enc + 1)
        alpha_t = torch.sum(phi_t, dim=1)
        alpha_t = alpha_t[:, 1:] - alpha_t[:, :-1]
        alpha_t[alpha_t == 0] = self.eps
        # print("alpha_t: ", alpha_t.size())
        # Apply masking
        if mask is not None:
            alpha_t.data.masked_fill_(mask, self.score_mask_value)

        context = torch.bmm(alpha_t.unsqueeze(1), memory).squeeze(1)
        if memory_pitch is not None:
            context_pitch = torch.bmm(alpha_t.unsqueeze(1), memory_pitch).squeeze(1)

        self.mu_prev = mu_cur
        
        if memory_pitch is not None:
            return context, context_pitch, alpha_t
        return context, alpha_t

