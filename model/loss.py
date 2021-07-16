import torch.nn as nn

class binary_cross_entropy():

    def __init__(self, ignore_index = ''):
        self.fn = nn.CrossEntropyLoss(ignore_index = ignore_index)

    def __call__(self, target, pred):
        """ 
        Binary Cross Entropy between the target and the output prediction.
        target: [B  ]
        pred: [B, N_class]
        """
        return self.fn(target, pred)
