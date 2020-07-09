import torch
from pytorch_pretrained_bert import BertModel, BertConfig, BertTokenizer

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# print(tokenizer.tokenize("Hello, my dog is cute"))
# input_ids = torch.tensor(tokenizer.tokenize("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids)
# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

indtoken = [[2384, 2260, 2431, 1173, 1877,    2, 1984,    2, 1480, 2202, 2392,    9,
                      2384,  717, 2202, 2058, 1480, 2202, 1251,  103,  160, 2202,  316, 2300,
                      1819,    9, 2300, 1819,   80,  103, 2384, 1165, 2202,  246,    9, 2104,
                      2430, 2496,  791, 2202, 1842,    9,    0,    0,    0,    0,    0,    0,
                         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                         0,    0,    0,    0,    0,    0,    0,    0]]
indtoken = torch.tensor(indtoken)[0]
# print(torch.tensor(indtoken))
# raise
import json
with open("index_to_word.json") as f:
    data = list(json.load(f)["word"].keys())
    print(data)
    l = list(map(lambda ind: data[ind], indtoken))
    print(l)

indexed_tokens = tokenizer.convert_tokens_to_ids(l)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])


# Load pre-trained model (weights)
# configuration = BertConfig(hidden_size=128,
#                            vocab_size_or_config_json_file=30522,
#                            num_attention_heads=8,
#                            )
# model = BertModel(configuration)
model = BertModel.from_pretrained('bert-base-uncased')


model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

# Predict hidden states features for each layer
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor)
# We have a hidden states for each of the 12 layers in model bert-base-uncased
assert len(encoded_layers) == 12
print(len(encoded_layers))
print(encoded_layers[1].size())
print(encoded_layers[2].size())