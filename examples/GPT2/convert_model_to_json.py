import torch
from torch.utils.data import Dataset

from json import JSONEncoder
import json

class EncodeTensor(JSONEncoder,Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(NpEncoder, self).default(obj)

model = torch.load("./models/gpt-nano/model.pt", map_location=torch.device("cpu"))
with open("./models/gpt-nano/model_weights.json", 'w') as json_file:
    json.dump(model, json_file,cls=EncodeTensor)
