import torch.nn as nn
import torch.nn.functional as F

class PreNet(nn.Module):
    def __init__(self, in_dims, fc1_dims=256, fc2_dims=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.p = dropout

    def forward(self, x):
        """forward

        Args:
            x (3D tensor with size `[batch_size, num_chars, tts_embed_dims]`): input texts list

        Returns:
            3D tensor with size `[batch_size, num_chars, encoder_dims]`
            
        """        
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=True)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=True)
        return x
