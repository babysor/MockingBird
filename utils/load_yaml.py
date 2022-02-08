import yaml


def load_hparams(filename):
    stream = open(filename, 'r')
    docs = yaml.safe_load_all(stream)
    hparams_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            hparams_dict[k] = v
    return hparams_dict

def merge_dict(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_dict(user[k], v)
    return user

class Dotdict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value

class HpsYaml(Dotdict):
    def __init__(self, yaml_file):
        super(Dotdict, self).__init__()
        hps = load_hparams(yaml_file)
        hp_dict = Dotdict(hps)
        for k, v in hp_dict.items():
            setattr(self, k, v)

    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__
    






