import config
import torch
from torch.nn import nn

class DeBERTaDataset:
    def __init__(self, text, target, max_len):
        self.text = text
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER


    def __len__(self):
        return self.target.shape[0]
        

    def __getitem__(self, item):
        text = " ".join(str(self.text[item]).split())
        target = self.target[item]

        inputs = self.tokenizer(
            text,
            return_tensor = 'pt',
            max_len = self.max_len,
            padding = max_len,
            truncation = True
            add_special_tokens = True,
            max_length = self.max_len
        )


        return {
            'inputs': {'input_ids': inputs[input_ids][0],
                       'attention_mask': inputs[attention_mask][0],
                       'token_type_ids': inputs[token_type_ids][0],
            'target': torch.tensor(target, dtype=torch.float)
        }