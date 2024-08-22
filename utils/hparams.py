import yaml
import json
import ast

def load_hparams_json(filename):
    with open(filename, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def load_hparams_yaml(filename):
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
        hps = load_hparams_yaml(yaml_file)
        hp_dict = Dotdict(hps)
        for k, v in hp_dict.items():
            setattr(self, k, v)

    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__
    
class HParams():
    def __init__(self, **kwargs): 
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v
    def keys(self):
        return self.__dict__.keys()
    def __setitem__(self, key, value): setattr(self, key, value)
    def __getitem__(self, key): return getattr(self, key)
    def keys(self):  return self.__dict__.keys()
    def items(self): return self.__dict__.items()
    def values(self): return self.__dict__.values()
    def __contains__(self, key): return key in self.__dict__
    def __repr__(self):
        return self.__dict__.__repr__()

    def parse(self, string):
        # Overrides hparams from a comma-separated string of name=value pairs
        if len(string) > 0:
            overrides = [s.split("=") for s in string.split(",")]
            keys, values = zip(*overrides)
            keys = list(map(str.strip, keys))
            values = list(map(str.strip, values))
            for k in keys:
                self.__dict__[k] = ast.literal_eval(values[keys.index(k)])
        return self

    def loadJson(self, fpath):
        with fpath.open("r", encoding="utf-8") as f:
            print("\Loading the json with %s\n", fpath)
            data = json.load(f)
            for k in data.keys():
                if k not in ["tts_schedule", "tts_finetune_layers"]: 
                    v = data[k]
                    if type(v) == dict:
                        v = HParams(**v)
                    self.__dict__[k] = v
        return self

    def dumpJson(self, fp):
        print("\Saving the json with %s\n", fp)
        with fp.open("w", encoding="utf-8") as f:
            json.dump(self.__dict__, f)
        return self



