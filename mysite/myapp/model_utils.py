import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

attributes = ['Rating_1', 'Rating_2', 'Rating_3', 'Rating_4', 'Rating_7', 'Rating_8', 'Rating_9', 'Rating_10']

config = {
    'model_name': 'roberta-base',
    'attributes': attributes,
    'lr': 1e-6,
    'weigth_decay': 1e-3,
    'n_epochs': 10,
}

tokenizer = AutoTokenizer.from_pretrained(config['model_name'], use_fast=True)

class ACLIMDB_Rating_Classifier(nn.Module):
    def __init__(self, config):
        super(ACLIMDB_Rating_Classifier, self).__init__()
        self.config = config
        self.pretrained_model = AutoModel.from_pretrained(self.config['model_name'])
        self.h_layer = nn.Linear(in_features = self.pretrained_model.config.hidden_size,
                       out_features = self.pretrained_model.config.hidden_size)
        self.c_head = nn.Linear(in_features = self.pretrained_model.config.hidden_size,
                      out_features = len(self.config['attributes']))
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
    def forward(self, input_ids, attention_mask):
        output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        tensor_out = torch.mean(output.last_hidden_state, 1)
        tensor_out = self.dropout(tensor_out)
        tensor_out = self.h_layer(tensor_out)
        tensor_out = self.relu(tensor_out)
        tensor_out = self.dropout(tensor_out)
        logits = self.c_head(tensor_out)
        return logits
    
model = ACLIMDB_Rating_Classifier(config)

checkpoint = torch.load('C:\\Users\\ilyak\\OneDrive\\Desktop\\RosatomIntership\\project\\model_data\\best_model_CEloss.pt')

model.load_state_dict(checkpoint['state_dict'], strict=False)

def model_prediction(user_input: str):
    tokens = tokenizer.encode_plus(
        user_input,
        max_length=256,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        return_attention_mask=True
    )

    model.eval()

    input_ids, attention_mask = tokens.input_ids, tokens.attention_mask
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        output = nn.functional.softmax(output, dim=1).numpy()
        print(output)
        output_argmax = np.argmax(output)
    
    return output_argmax, 1 if output_argmax >= 4 else 0