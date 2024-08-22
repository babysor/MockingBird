"""VGG2L definition for transformer-transducer."""

import torch


class VGG2L(torch.nn.Module):
    """VGG2L module for transformer-transducer encoder."""

    def __init__(self, idim, odim):
        """Construct a VGG2L object.

        Args:
            idim (int): dimension of inputs
            odim (int): dimension of outputs

        """
        super(VGG2L, self).__init__()

        self.vgg2l = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((3, 2)),
            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
        )

        self.output = torch.nn.Linear(128 * ((idim // 2) // 2), odim)

    def forward(self, x, x_mask):
        """VGG2L forward for x.

        Args:
            x (torch.Tensor): input torch (B, T, idim)
            x_mask (torch.Tensor): (B, 1, T)

        Returns:
            x (torch.Tensor): input torch (B, sub(T), attention_dim)
            x_mask (torch.Tensor): (B, 1, sub(T))

        """
        x = x.unsqueeze(1)
        x = self.vgg2l(x)

        b, c, t, f = x.size()

        x = self.output(x.transpose(1, 2).contiguous().view(b, t, c * f))

        if x_mask is None:
            return x, None
        else:
            x_mask = self.create_new_mask(x_mask, x)

            return x, x_mask

    def create_new_mask(self, x_mask, x):
        """Create a subsampled version of x_mask.

        Args:
            x_mask (torch.Tensor): (B, 1, T)
            x (torch.Tensor): (B, sub(T), attention_dim)

        Returns:
            x_mask (torch.Tensor): (B, 1, sub(T))

        """
        x_t1 = x_mask.size(2) - (x_mask.size(2) % 3)
        x_mask = x_mask[:, :, :x_t1][:, :, ::3]

        x_t2 = x_mask.size(2) - (x_mask.size(2) % 2)
        x_mask = x_mask[:, :, :x_t2][:, :, ::2]

        return x_mask
