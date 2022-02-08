import torch


def gcd(a, b):  
    """Greatest common divisor."""
    a, b = (a, b) if a >=b else (b, a)
    if a%b == 0:  
        return b  
    else :  
        return gcd(b, a%b) 

def lcm(a, b):
    """Least common multiple"""
    return a * b // gcd(a, b)

def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

