import torch
import torch.nn as nn
from transformers import BertForMultipleChoice
import random

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LinearClassifier(BertForMultipleChoice):
    def __init__(self,config):
        super().__init__(config)

        self.config=config

        self.classifier=nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,20),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(0.2),
            nn.Linear(20,1)
        )

        self.init_weights()

    def create_pooled_text_embeddings(self,batch_inputs):
        num_options=batch_inputs["input_ids"].size(0)
        ret=torch.empty(num_options,self.config.hidden_size).to(device)

        for i in range(num_options):
            option_inputs={
                "input_ids":batch_inputs["input_ids"][i].unsqueeze(0),
                "attention_mask":batch_inputs["attention_mask"][i].unsqueeze(0),
                "token_type_ids":batch_inputs["token_type_ids"][i].unsqueeze(0),
            }

            outputs=self.bert(**option_inputs)
            ret[i]=outputs[1][0]

        return ret

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        batch_size=input_ids.size(0)
        num_options=input_ids.size(1)

        text_embeddings=torch.empty(batch_size,num_options,self.config.hidden_size).to(device)
        for i in range(batch_size):
            batch_inputs={
                "input_ids":input_ids[i],
                "attention_mask":attention_mask[i],
                "token_type_ids":token_type_ids[i]
            }

            text_embeddings[i]=self.create_pooled_text_embeddings(batch_inputs)

        text_embeddings=text_embeddings.view(-1,self.config.hidden_size)
        logits=self.classifier(text_embeddings)
        reshaped_logits=logits.view(-1,num_options)

        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(reshaped_logits, labels)

        return (loss,reshaped_logits)

class ConvClassifier(BertForMultipleChoice):
    def __init__(self,config):
        super().__init__(config)

        self.config=config

        self.conv=nn.Sequential(
            nn.Conv2d(1,32,8,4,2),  #(N,32,128,192)
            nn.LeakyReLU(0.2),

            nn.Conv2d(32,64,8,4,2), #(N,64,32,48)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64,128,8,4,2), #(N,128,8,12)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,256,8,4,2), #(N,256,2,3)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256,512,(2,3),1,0), #(N,512,1,1)
        )
        self.classifier=(
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,100),
            nn.LeakyReLU(0.2),
            nn.Linear(100,1)
        )

        self.init_weights()

    def create_text_embeddings(self,batch_inputs):
        num_options=batch_inputs["input_ids"].size(0)
        ret=torch.empty(num_options,512,768).to(device)

        for i in range(num_options):
            option_inputs={
                "input_ids":batch_inputs["input_ids"][i].unsqueeze(0),
                "attention_mask":batch_inputs["attention_mask"][i].unsqueeze(0),
                "token_type_ids":batch_inputs["token_type_ids"][i].unsqueeze(0),
            }

            outputs=self.bert(**option_inputs)
            ret[i]=outputs[0][0]

        return ret

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        batch_size=input_ids.size(0)
        num_options=input_ids.size(1)

        text_embeddings=torch.empty(batch_size,num_options,512,768).to(device)
        for i in range(batch_size):
            batch_inputs={
                "input_ids":input_ids[i],
                "attention_mask":attention_mask[i],
                "token_type_ids":token_type_ids[i]
            }

            text_embeddings[i]=self.create_text_embeddings(batch_inputs)

        text_embeddings=text_embeddings.view(-1,1,512,768)  #(N*num_options,1,512,768)
        logits=self.conv(text_embeddings)   #(N*num_options,512,1,1)
        logits=torch.squeeze(logits)    #(N*num_options,512)
        logits=self.classifier(logits)  #(N*num_options,1)
        reshaped_logits=logits.view(-1,num_options) #(N,num_options)

        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(reshaped_logits, labels)

        return (loss,reshaped_logits)
