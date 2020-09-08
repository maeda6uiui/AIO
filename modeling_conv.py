import argparse
import logging
import os
import sys
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader,TensorDataset
from transformers import AdamW,get_linear_schedule_with_warmup

from models import ConvClassifier

#Fix the seed.
SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

#Set up a logger.
logging_fmt = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_dataset(input_dir,num_examples=-1,num_options=4):
    input_ids=torch.load(os.path.join(input_dir,"input_ids.pt"))
    attention_mask=torch.load(os.path.join(input_dir,"attention_mask.pt"))
    token_type_ids=torch.load(os.path.join(input_dir,"token_type_ids.pt"))
    labels=torch.load(os.path.join(input_dir,"labels.pt"))

    input_ids=input_ids[:,:num_options,:]
    attention_mask=attention_mask[:,:num_options,:]
    token_type_ids=token_type_ids[:,:num_options,:]

    if num_examples>0:
        input_ids=input_ids[:num_examples,:,:]
        attention_mask=attention_mask[:num_examples,:,:]
        token_type_ids=token_type_ids[:num_examples,:,:]
        labels=labels[:num_examples]

    return TensorDataset(input_ids,attention_mask,token_type_ids,labels)

def train(classifier_model,optimizer,scheduler,dataloader):
    classifier_model.train()

    count_steps=0
    total_loss=0

    for batch_idx,batch in enumerate(dataloader):
        batch = tuple(t for t in batch)
        batch_size=batch[0].size(0)
        bert_inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[1].to(device),
            "token_type_ids": batch[2].to(device),
            "labels": batch[3].to(device),
        }

        base_index=1
        for i in range(7):
            if i==6:
                option_indices=[0,base_index]
            else:
                option_indices=[0,base_index,base_index+1,base_index+2]
                base_index+=3

            i_input_ids=torch.empty(batch_size,len(option_indices),512,dtype=torch.long).to(device)
            i_attention_mask=torch.empty(batch_size,len(option_indices),512,dtype=torch.long).to(device)
            i_token_type_ids=torch.empty(batch_size,len(option_indices),512,dtype=torch.long).to(device)
            i_labels=torch.empty(batch_size,dtype=torch.long).to(device)

            for j in range(batch_size):
                for k,index in enumerate(option_indices):
                    i_input_ids[j,k]=bert_inputs["input_ids"][j,index]
                    i_attention_mask[j,k]=bert_inputs["attention_mask"][j,index]
                    i_token_type_ids[j,k]=bert_inputs["token_type_ids"][j,index]
                i_labels[j]=bert_inputs["labels"][j]

            i_bert_inputs={
                "input_ids":i_input_ids,
                "attention_mask":i_attention_mask,
                "token_type_ids":i_token_type_ids,
                "labels":i_labels
            }

            # Initialize gradiants
            optimizer.zero_grad()
            # Forward propagation
            classifier_outputs=classifier_model(**i_bert_inputs)
            loss=classifier_outputs[0]
            # Backward propagation
            loss.backward()
            # Update parameters
            optimizer.step()
            scheduler.step()

            count_steps+=1
            total_loss+=loss.item()

        if batch_idx%100==0:
            logger.info("Step: {}\tLoss: {}\tlr: {}".format(
                batch_idx,loss.item(),optimizer.param_groups[0]["lr"]))

    return total_loss/count_steps

def simple_accuracy(pred_labels, correct_labels):
    return (pred_labels == correct_labels).mean()

def evaluate(classifier_model,dataloader):
    classifier_model.eval()

    preds = None
    correct_labels = None
    for batch_idx,batch in tqdm(enumerate(dataloader),total=len(dataloader)):
        with torch.no_grad():
            batch = tuple(t for t in batch)
            bert_inputs = {
                "input_ids": batch[0].to(device),
                "attention_mask": batch[1].to(device),
                "token_type_ids": batch[2].to(device),
                "labels": batch[3].to(device),
            }

            classifier_outputs=classifier_model(**bert_inputs)
            logits=classifier_outputs[1]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                correct_labels = bert_inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                correct_labels = np.append(
                    correct_labels, bert_inputs["labels"].detach().cpu().numpy(), axis=0
                )

    pred_labels = np.argmax(preds, axis=1)
    accuracy = simple_accuracy(pred_labels, correct_labels)

    return pred_labels,correct_labels,accuracy

def main(batch_size,num_epochs,lr,train_input_dir,dev1_input_dir,result_save_dir):
    logger.info("batch_size: {} num_epochs: {} lr: {}".format(batch_size,num_epochs,lr))

    #Create dataloaders.
    train_dataset=create_dataset(train_input_dir,num_examples=-1,num_options=20)
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

    dev1_dataset=create_dataset(dev1_input_dir,num_examples=-1,num_options=20)
    dev1_dataloader=DataLoader(dev1_dataset,batch_size=4,shuffle=False,drop_last=True)

    #Create a classifier model.
    logger.info("Create a classifier model.")
    classifier_model=ConvClassifier.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    classifier_model.to(device)

    #Create an optimizer and a scheduler.
    optimizer=AdamW(classifier_model.parameters(),lr=lr,eps=1e-8)
    total_steps = len(train_dataloader)*7*num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    #Create a directory to save the results in.
    os.makedirs(result_save_dir,exist_ok=True)

    logger.info("Start model training.")
    for epoch in range(num_epochs):
        logger.info("===== Epoch {}/{} =====".format(epoch+1,num_epochs))

        mean_loss=train(classifier_model,optimizer,scheduler,train_dataloader)
        logger.info("Mean loss: {}".format(mean_loss))

        #Save model parameters.
        checkpoint_filepath=os.path.join(result_save_dir,"checkpoint_{}.pt".format(epoch+1))
        torch.save(classifier_model.state_dict(),checkpoint_filepath)

        pred_labels,correct_labels,accuracy=evaluate(classifier_model,dev1_dataloader)
        logger.info("Accuracy: {}".format(accuracy))

        #Save results as text files.
        res_filepath=os.path.join(result_save_dir,"result_eval_{}.txt".format(epoch+1))
        labels_filepath=os.path.join(result_save_dir,"labels_eval_{}.txt".format(epoch+1))

        with open(res_filepath,"w") as w:
            w.write("Accuracy: {}\n".format(accuracy))

        with open(labels_filepath,"w") as w:
            for pred_label,correct_label in zip(pred_labels,correct_labels):
                w.write("{} {}\n".format(pred_label,correct_label))

    logger.info("Finished model training.")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="AIO")

    parser.add_argument("--batch_size",type=int,default=4)
    parser.add_argument("--num_epochs",type=int,default=10)
    parser.add_argument("--lr",type=float,default=5e-5)
    parser.add_argument("--train_input_dir",type=str,default="~/EncodedCache/Train")
    parser.add_argument("--dev1_input_dir",type=str,default="~/EncodedCache/Dev1")
    parser.add_argument("--result_save_dir",type=str,default="./OutputDir/Conv")

    args=parser.parse_args()

    main(
        args.batch_size,
        args.num_epochs,
        args.lr,
        args.train_input_dir,
        args.dev1_input_dir,
        args.result_save_dir
    )
