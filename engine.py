from tqdm import tqdm
import torch
from torch.nn import nn


device = torch.device('cpu')

def loss_fn(outputs, target):
    return nn.BCEWithLogitsLoss()(outputs, target(-1, 1))

def train_fn(data_loader, model, optimizer, device):
    model_train()
 
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d['ids']
        mask = d['mask']
        token_type_ids = d['token_type_ids']
        score = d['score']

        # loading the train_data into device(CPU)

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        score = score.to(device, dtype=torch.float)

        outputs = model(
            ids=ids,
            mask= mask,
            token_type_ids=token_type_ids
        )


        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

def eval_fn(data_loader, model, device):
    model.eval()
    final_score = []
    final_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_ids = d['input_ids']
            attention_mask = d['attention_mask']
            token_type_ids = d['token_type_ids']
            target = d['target']

        # loading the train_data into device(GPU)

            input_ids = input_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            target = target.to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids
            )

            final_score.extend(score.cpu().detach().numpy().to_list())
            final_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().to_list())

    return final_score, final_outputs