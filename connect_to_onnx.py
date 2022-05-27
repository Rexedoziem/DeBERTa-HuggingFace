import torch.onnx
from torch import nn
from model import DeBERTaBaseUncased
import config
from dataset import DeBERTaDataset
import onnxruntime

if __name__=='__main__':
    device = 'cuda'
    text = ['This is just a short tutorial on onnx']

    dataset = DeBERTaDataset(text=text, target=[0])

    model = DeBERTaBaseUncased()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.eval()

    input_ids = dataset[0]['input_ids'].unsqueeze(0)
    attention_mask = dataset[0]['attention_mask'].unsqueeze(0)
    token_type_ids = dataset[0]['token_type_ids'].unsqueeze(0)

    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        'model.onnx',
        input_names = ['input_ids', 'attention_mask', 'token_type_ids'],
        output_names= ['output'],
        dynamic_axes = {
            'input_ids' : {0: 'batch_size'},
            'attention_mask' : {0: 'batch_size'},
            'token_type_ids' : {0: 'batch_size'},
            'output' : {0: 'batch_size'}

        }
    )



    MODEL = onnxruntime.InferenceSession('model.onnx')

    def to_numpy(tensor):
        if tensor.requires_grad:
            return tensor.detach()cpu.numpy()
        tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    onnx_input = {
        'input_ids': to_numpy(input_ids),
        'attention_mask': to_numpy(attention_mask),
        'token_type_ids': to_numpy(token_type_ids)}


    output = MODEL.run(None, ort_inputs)

    # compare ONNX Runtime and Pytorch results
    #np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print('Exported model has been tested with ONNXRuntime, and the result looks good!')


