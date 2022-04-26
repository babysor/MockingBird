import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from vocoder.fregan.utils import init_weights, get_padding

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5, 7)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[3],
                               padding=get_padding(kernel_size, dilation[3])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class FreGAN(torch.nn.Module):
    def __init__(self, h, top_k=4):
        super(FreGAN, self).__init__()
        self.h = h

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.upsample_rates = h.upsample_rates
        self.up_kernels = h.upsample_kernel_sizes
        self.cond_level = self.num_upsamples - top_k
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        self.cond_up = nn.ModuleList()
        self.res_output = nn.ModuleList()
        upsample_ = 1
        kr = 80

        for i, (u, k) in enumerate(zip(self.upsample_rates, self.up_kernels)):
#            self.ups.append(weight_norm(
 #               ConvTranspose1d(h.upsample_initial_channel // (2 ** i), h.upsample_initial_channel // (2 ** (i + 1)),
 #                               k, u, padding=(k - u) // 2)))
            self.ups.append(weight_norm(ConvTranspose1d(h.upsample_initial_channel//(2**i),
                            h.upsample_initial_channel//(2**(i+1)),
                            k, u, padding=(u//2 + u%2), output_padding=u%2)))

            if i > (self.num_upsamples - top_k):
                self.res_output.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=u, mode='nearest'),
                        weight_norm(nn.Conv1d(h.upsample_initial_channel // (2 ** i),
                                              h.upsample_initial_channel // (2 ** (i + 1)), 1))
                    )
                )
            if i >= (self.num_upsamples - top_k):
                self.cond_up.append(
                    weight_norm(
                        ConvTranspose1d(kr, h.upsample_initial_channel // (2 ** i),
                                        self.up_kernels[i - 1], self.upsample_rates[i - 1],
                                        padding=(self.upsample_rates[i-1]//2+self.upsample_rates[i-1]%2), output_padding=self.upsample_rates[i-1]%2))
                )
                kr = h.upsample_initial_channel // (2 ** i)

            upsample_ *= u

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.cond_up.apply(init_weights)
        self.res_output.apply(init_weights)

    def forward(self, x):
        mel = x
        x = self.conv_pre(x)
        output = None
        for i in range(self.num_upsamples):
            if i >= self.cond_level:
                mel = self.cond_up[i - self.cond_level](mel)
                x += mel
            if i > self.cond_level:
                if output is None:
                    output = self.res_output[i - self.cond_level - 1](x)
                else:
                    output = self.res_output[i - self.cond_level - 1](output)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
            if output is not None:
                output = output + x

        x = F.leaky_relu(output)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        for l in self.cond_up:
            remove_weight_norm(l)
        for l in self.res_output:
            remove_weight_norm(l[1])
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


'''
    to run this, fix 
    from . import ResStack
    into
    from res_stack import ResStack
'''
if __name__ == '__main__':
    '''
    torch.Size([3, 80, 10])
    torch.Size([3, 1, 2000])
    4527362
    '''
    with open('config.json') as f:
        data = f.read()
    from utils import AttrDict
    import json
    json_config = json.loads(data)
    h = AttrDict(json_config)
    model = FreGAN(h)

    c = torch.randn(3, 80, 10)  # (B, channels, T).
    print(c.shape)

    y = model(c) # (B, 1, T ** prod(upsample_scales)
    print(y.shape)
    assert y.shape == torch.Size([3, 1, 2560])  # For normal melgan torch.Size([3, 1, 2560])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)