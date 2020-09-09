import torch
import torch.nn as nn
from transformers import BertForMultipleChoice

class LinearClassifier(BertForMultipleChoice):
    def __init__(self,config):
        super().__init__(config)

        self.classifier=nn.Sequential(
            #nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size,1),
        )

        self.init_weights()
