import config
import transformers
import torch.nn as nn
from DeBERTa import deberta


class DeBERTaBaseUncased(nn.Module):
    def __init__(self):
        super(DeBERTaBaseUncased, self).__init__()
        self.deberta = transformers.AutoTokenizer.from_pretrained(config.DeBERTa_PATH)
        self.deberta_drop = nn.Dropout(0.3)
        self.out = nn.linear(768, 1)

    
    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.deberta(
        ids,
        attention_mask = mask,
        token_type_ids = token_type_ids
        )
        return o2