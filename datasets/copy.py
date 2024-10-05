import torch
from torch.utils.data import Dataset
from torch.distributions.binomial import Binomial
import torch.nn.functional as F

class CopyDataset(Dataset):
    """A Dataset class to generate random examples for the copy task. Each
    sequence has a random length between `min_seq_len` and `max_seq_len`.
    Each vector in the sequence has a fixed length of `seq_width`. The vectors
    are bounded by start and end delimiter flags.

    To account for the delimiter flags, the input sequence length as well
    width is two more than the target sequence.
    """

    def __init__(self, task_params):
        """Initialize a dataset instance for copy task.

        Arguments
        ---------
        task_params : dict
            A dict containing parameters relevant to copy task.
        """
        self.seq_width = task_params['seq_width']
        self.min_seq_len = task_params['min_seq_len']
        self.max_seq_len = task_params['max_seq_len']
        self.in_dim = task_params['seq_width'] + 2
        self.out_dim = task_params['seq_width']
        self.prob = task_params['prob']
        self.mode = task_params['mode']

    def __len__(self):
        # sequences are generated randomly so this does not matter
        # set a sufficiently large size for data loader to sample mini-batches
        return 65536


    def get_sample_wlen(self, seq_len, bs=1):
        # idx only acts as a counter while generating batches.
        
        if self.mode == "onehot":
            seq = F.one_hot(torch.randint(self.seq_width, (seq_len, bs)),num_classes=self.seq_width)
        else:
            prob = self.prob * torch.ones([seq_len, bs, self.seq_width], dtype=torch.float64)
            seq = Binomial(1, prob).sample()
            
        # fill in input sequence, two bit longer and wider than target
        input_seq = torch.zeros([seq_len + 2, bs,self.seq_width + 2])
        input_seq[0, :,self.seq_width] = 1.0  # start delimiter
        input_seq[1:seq_len + 1,:, :self.seq_width] = seq
        input_seq[seq_len + 1, :, self.seq_width + 1] = 1.0  # end delimiter

        target_seq = torch.zeros([seq_len, bs, self.seq_width])
        target_seq[:seq_len,:, :self.seq_width] = seq
        return {'input': input_seq, 'target': target_seq}

