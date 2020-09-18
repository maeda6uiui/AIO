import torch
import torch.nn as nn
from transformers import BertForMultipleChoice

class LinearClassifier(BertForMultipleChoice):
    def __init__(self,config):
        super().__init__(config)

        self.classifier=nn.Sequential(
            nn.Linear(config.hidden_size,1)
        )

        self.init_weights()

class DoubleLinearClassifier(nn.Module):
    def __init__(self,num_options=20):
        super().__init__()

        self.classifier=nn.Sequential(
            nn.Linear(2*num_options,2*num_options),
            nn.LeakyReLU(0.2),
            nn.Linear(2*num_options,2*num_options),
            nn.LeakyReLU(0.2),
            nn.Linear(2*num_options,num_options)
        )

    def forward(self,logits_1,logits_2):
        logits=torch.cat([logits_1,logits_2],dim=1)
        return self.classifier(logits)
