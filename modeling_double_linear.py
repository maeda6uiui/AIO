import argparse
import logging
import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader,TensorDataset
from transformers import BertConfig

from models import LinearClassifier,DoubleLinearClassifier

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

def train(classifier_model,lc_model_1,lc_model_2,optimizer,dataloader_1,dataloader_2):
    classifier_model.train()
    lc_model_1.eval()
    lc_model_2.eval()

    criterion=nn.CrossEntropyLoss()

    count_steps=0
    total_loss=0

    for batch_idx,(batch_1,batch_2) in enumerate(zip(dataloader_1,dataloader_2)):
        batch_1 = tuple(t for t in batch_1)
        bert_inputs_1 = {
            "input_ids": batch_1[0].to(device),
            "attention_mask": batch_1[1].to(device),
            "token_type_ids": batch_1[2].to(device),
            "labels": batch_1[3].to(device),
        }

        batch_2 = tuple(t for t in batch_2)
        bert_inputs_2 = {
            "input_ids": batch_2[0].to(device),
            "attention_mask": batch_2[1].to(device),
            "token_type_ids": batch_2[2].to(device),
            "labels": batch_2[3].to(device),
        }

        with torch.no_grad():
            lc1_outputs=lc_model_1(**bert_inputs_1)
            lc2_outputs=lc_model_2(**bert_inputs_2)

        # Initialize gradiants
        optimizer.zero_grad()
        # Forward propagation
        classifier_outputs=classifier_model(lc1_outputs[1],lc2_outputs[1])
        loss=criterion(classifier_outputs,bert_inputs_1["labels"])
        # Backward propagation
        loss.backward()
        # Update parameters
        optimizer.step()

        count_steps+=1
        total_loss+=loss.item()

        if batch_idx%100==0:
            logger.info("Step: {}\tLoss: {}\tlr: {}".format(
                batch_idx,loss.item(),optimizer.param_groups[0]["lr"]))

    return total_loss/count_steps

def simple_accuracy(pred_labels, correct_labels):
    return (pred_labels == correct_labels).mean()

def evaluate(classifier_model,lc_model_1,lc_model_2,dataloader_1,dataloader_2):
    classifier_model.eval()
    lc_model_1.eval()
    lc_model_2.eval()

    preds = None
    correct_labels = None
    for batch_idx,(batch_1,batch_2) in tqdm(enumerate(zip(dataloader_1,dataloader_2)),total=len(dataloader_1)):
        with torch.no_grad():
            batch_1 = tuple(t for t in batch_1)
            bert_inputs_1 = {
                "input_ids": batch_1[0].to(device),
                "attention_mask": batch_1[1].to(device),
                "token_type_ids": batch_1[2].to(device),
                "labels": batch_1[3].to(device),
            }

            batch_2 = tuple(t for t in batch_2)
            bert_inputs_2 = {
                "input_ids": batch_2[0].to(device),
                "attention_mask": batch_2[1].to(device),
                "token_type_ids": batch_2[2].to(device),
                "labels": batch_2[3].to(device),
            }

            lc1_outputs=lc_model_1(**bert_inputs_1)
            lc2_outputs=lc_model_2(**bert_inputs_2)

            logits=classifier_model(lc1_outputs[1],lc2_outputs[1])

            if preds is None:
                preds = logits.detach().cpu().numpy()
                correct_labels = bert_inputs_1["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                correct_labels = np.append(
                    correct_labels, bert_inputs_1["labels"].detach().cpu().numpy(), axis=0
                )

    pred_labels = np.argmax(preds, axis=1)
    accuracy = simple_accuracy(pred_labels, correct_labels)

    return pred_labels,correct_labels,accuracy

def main(
    batch_size,num_epochs,lr,
    bert_model_dir_1,finetuned_model_filepath_1,train_input_dir_1,dev1_input_dir_1,
    bert_model_dir_2,finetuned_model_filepath_2,train_input_dir_2,dev1_input_dir_2,
    result_save_dir):
    logger.info("batch_size: {} num_epochs: {} lr: {}".format(batch_size,num_epochs,lr))

    #Create dataloaders.
    logger.info("Create train dataloader from {}.".format(train_input_dir_1))
    train_dataset_1=create_dataset(train_input_dir_1,num_examples=-1,num_options=20)
    train_dataloader_1=DataLoader(train_dataset_1,batch_size=batch_size,shuffle=True,drop_last=True)

    logger.info("Create train dataloader from {}.".format(train_input_dir_2))
    train_dataset_2=create_dataset(train_input_dir_2,num_examples=-1,num_options=20)
    train_dataloader_2=DataLoader(train_dataset_2,batch_size=batch_size,shuffle=True,drop_last=True)

    logger.info("Create dev1 dataloader from {}.".format(dev1_input_dir_1))
    dev1_dataset_1=create_dataset(dev1_input_dir_1,num_examples=-1,num_options=20)
    dev1_dataloader_1=DataLoader(dev1_dataset_1,batch_size=4,shuffle=False,drop_last=True)

    logger.info("Create dev1 dataloader from {}.".format(dev1_input_dir_2))
    dev1_dataset_2=create_dataset(dev1_input_dir_2,num_examples=-1,num_options=20)
    dev1_dataloader_2=DataLoader(dev1_dataset_2,batch_size=4,shuffle=False,drop_last=True)

    #Create a classifier model.
    classifier_model=DoubleLinearClassifier(20)
    classifier_model.to(device)

    lc_model_1=None
    if bert_model_dir_1=="USE_DEFAULT":
        lc_model_1=LinearClassifier.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    else:
        config_filepath=os.path.join(bert_model_dir_1,"bert_config.json")
        bert_model_filepath=os.path.join(bert_model_dir_1,"pytorch_model.bin")

        bert_config=BertConfig.from_pretrained(bert_model_dir_1)
        logger.info(bert_config)
        lc_model_1=LinearClassifier.from_pretrained(bert_model_filepath,config=bert_config)
    
    lc_model_2=None
    if bert_model_dir_2=="USE_DEFAULT":
        lc_model_2=LinearClassifier.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    else:
        config_filepath=os.path.join(bert_model_dir_2,"bert_config.json")
        bert_model_filepath=os.path.join(bert_model_dir_2,"pytorch_model.bin")

        bert_config=BertConfig.from_pretrained(bert_model_dir_2)
        logger.info(bert_config)
        lc_model_2=LinearClassifier.from_pretrained(bert_model_filepath,config=bert_config)

    logger.info("Load model parameters from {} and {}.".format(
        finetuned_model_filepath_1,finetuned_model_filepath_2))
    lc_model_1.load_state_dict(torch.load(finetuned_model_filepath_1,map_location=device))
    lc_model_2.load_state_dict(torch.load(finetuned_model_filepath_2,map_location=device))

    #Create an optimizer.
    optimizer=optim.AdamW(classifier_model.parameters(),lr=lr)
    
    #Create a directory to save the results in.
    os.makedirs(result_save_dir,exist_ok=True)

    logger.info("Start model training.")
    for epoch in range(num_epochs):
        logger.info("===== Epoch {}/{} =====".format(epoch+1,num_epochs))

        mean_loss=train(classifier_model,lc_model_1,lc_model_2,optimizer,train_dataloader_1,train_dataloader_2)
        logger.info("Mean loss: {}".format(mean_loss))

        #Save model parameters.
        checkpoint_filepath=os.path.join(result_save_dir,"checkpoint_{}.pt".format(epoch+1))
        torch.save(classifier_model.state_dict(),checkpoint_filepath)

        pred_labels,correct_labels,accuracy=evaluate(classifier_model,lc_model_1,lc_model_2,dev1_dataloader_1,dev1_dataloader_2)
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
    parser.add_argument("--bert_model_dir_1",type=str,default="USE_DEFAULT")
    parser.add_argument("--finetuned_model_filepath_1",type=str,default="~/AIO/WorkingDir/OutputDir/Linear/checkpoint_3.pt")
    parser.add_argument("--train_input_dir_1",type=str,default="~/EncodedCache/Train")
    parser.add_argument("--dev1_input_dir_1",type=str,default="~/EncodedCache/Dev1")
    parser.add_argument("--bert_model_dir_2",type=str,default="~/BERTModels/NICT_BERT-base_JapaneseWikipedia_100K")
    parser.add_argument("--finetuned_model_filepath_2",type=str,default="~/AIO/WorkingDir/OutputDir/NICTBERT/checkpoint_9.pt")
    parser.add_argument("--train_input_dir_2",type=str,default="~/EncodedCacheNICT2/Train")
    parser.add_argument("--dev1_input_dir_2",type=str,default="~/EncodedCacheNICT2/Dev1")
    parser.add_argument("--result_save_dir",type=str,default="./OutputDir/DoubleLinear")

    args=parser.parse_args()

    main(
        args.batch_size,
        args.num_epochs,
        args.lr,
        args.bert_model_dir_1,
        args.finetuned_model_filepath_1,
        args.train_input_dir_1,
        args.dev1_input_dir_1,
        args.bert_model_dir_2,
        args.finetuned_model_filepath_2,
        args.train_input_dir_2,
        args.dev1_input_dir_2,
        args.result_save_dir
    )
