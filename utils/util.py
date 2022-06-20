import matplotlib
matplotlib.use('Agg')
import time

class Timer():
    ''' Timer for recording training time distribution. '''
    def __init__(self):
        self.prev_t = time.time()
        self.clear()

    def set(self):
        self.prev_t = time.time()

    def cnt(self, mode):
        self.time_table[mode] += time.time()-self.prev_t
        self.set()
        if mode == 'bw':
            self.click += 1

    def show(self):
        total_time = sum(self.time_table.values())
        self.time_table['avg'] = total_time/self.click
        self.time_table['rd'] = 100*self.time_table['rd']/total_time
        self.time_table['fw'] = 100*self.time_table['fw']/total_time
        self.time_table['bw'] = 100*self.time_table['bw']/total_time
        msg = '{avg:.3f} sec/step (rd {rd:.1f}% | fw {fw:.1f}% | bw {bw:.1f}%)'.format(
            **self.time_table)
        self.clear()
        return msg

    def clear(self):
        self.time_table = {'rd': 0, 'fw': 0, 'bw': 0}
        self.click = 0

# Reference : https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/e2e_asr.py#L168

def human_format(num):
    magnitude = 0
    while num >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '{:3.1f}{}'.format(num, [' ', 'K', 'M', 'G', 'T', 'P'][magnitude])


# provide easy access of attribute from dict, such abc.key 
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
