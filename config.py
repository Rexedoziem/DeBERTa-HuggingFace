import transformers
from transformers import AutoTokenizer

class config:
    num_workers = 4
    DeBERTa_PATH = "../input/microsoft/deberta-base/"
    config_path = DeBERTa_PATH + 'config.path'
    scheduler = 'cosine'
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 4
    MODEL_PATH = "model.bin"
    TRAINING_FILE = "../input/train.csv"
    TOKENIZER =transformers.AutoTokenizer.from_pretrained(
        DeBERTa_PATH,
        do_lower_case = True
    )
