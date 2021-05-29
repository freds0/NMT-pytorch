from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field
from util.tokenizers import tokenize_de, tokenize_en


class Multi30kData:

    def  __init__(self):

        self.SRC = Field(tokenize = tokenize_de,
                    init_token = '<sos>',
                    eos_token = '<eos>',
                    lower = True)

        self.TRG = Field(tokenize = tokenize_en,
                    init_token = '<sos>',
                    eos_token = '<eos>',
                    lower = True)

    def splits(self):

        train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (self.SRC, self.TRG))

        return train_data, valid_data, test_data, self.SRC, self.TRG