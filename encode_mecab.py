import argparse
import gzip
import json
import logging
import os
import sys
import torch
import jaconv
import re
from tqdm import tqdm
from transformers import BertTokenizer
import MeCab

#Set up a logger.
logging_fmt = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

NUM_OPTIONS=20

class InputExample(object):
    def __init__(self, qid, question, endings, label=None):
        self.qid = qid
        self.question = question
        self.endings = endings
        self.label = label

def load_examples(example_filepath):
    with open(example_filepath, "r", encoding="UTF-8") as r:
        lines = r.read().splitlines()

    examples = []
    for line in lines:
        data = json.loads(line)

        qid = data["qid"]
        question = data["question"].replace("_", "")
        options = data["answer_candidates"][:NUM_OPTIONS]
        answer = data["answer_entity"]

        label=0
        if answer!="":
            label=options.index(answer)

        example = InputExample(qid, question, options, label)
        examples.append(example)

    return examples

def load_contexts(context_filepath):
    contexts={}

    with gzip.open(context_filepath,mode="rt",encoding="utf-8") as r:
        for line in r:
            data = json.loads(line)

            title=data["title"]
            text=data["text"]

            contexts[title]=text

    return contexts

def tokenize_with_mecab(mecab,text):
    #mecab.parse("")
    node=mecab.parseToNode(text)

    tokens=[]
    while node:
        word=node.surface
        tokens.append(word)

        node=node.next

    return tokens

def encode_plus(mecab,tokenizer,text_question,text_ending,text_context,max_length=512):
    #Question
    text_question=jaconv.h2z(text_question,kana=True,digit=True,ascii=True)
    tokens_question=tokenize_with_mecab(mecab,text_question)

    #Ending
    text_ending=jaconv.h2z(text_ending,kana=True,digit=True,ascii=True)
    tokens_ending=tokenize_with_mecab(mecab,text_ending)

    #Context
    text_context=jaconv.h2z(text_context,kana=True,digit=True,ascii=True)
    tokens_context=tokenize_with_mecab(mecab,text_context)

    tokens_a=["[CLS]"]+tokens_question+["[SEP]"]+tokens_ending+["[SEP]"]
    tokens_b=tokens_context+["[SEP]"]

    input_ids_a=tokenizer.convert_tokens_to_ids(tokens_a)
    input_ids_b=tokenizer.convert_tokens_to_ids(tokens_b)

    len_a=len(input_ids_a)
    len_b=len(input_ids_b)

    if len_a+len_b>max_length:
        input_ids_b=input_ids_b[:max_length-len_a]
        input_ids_b[max_length-len_a-1]=3   #[SEP]
    elif len_a+len_b<max_length:
        padding_length=max_length-(len_a+len_b)
        input_ids_b=input_ids_b+[0 for i in range(padding_length)]

    #Input IDs
    input_ids=input_ids_a+input_ids_b
    input_ids=torch.tensor(input_ids)

    #Attention mask
    attention_mask=torch.ones(max_length,dtype=torch.long)
    for i in range(len_a+len_b,max_length):
        attention_mask[i]=0

    #Token type IDs
    token_type_ids=torch.ones(max_length,dtype=torch.long)
    for i in range(len_a):
        token_type_ids[i]=0

    encoding={
        "input_ids":input_ids,
        "attention_mask":attention_mask,
        "token_type_ids":token_type_ids
    }

    return encoding

def encode_examples(mecab,tokenizer,examples,contexts,max_seq_length):
    input_ids=torch.empty(len(examples),NUM_OPTIONS,max_seq_length,dtype=torch.long)
    attention_mask=torch.empty(len(examples),NUM_OPTIONS,max_seq_length,dtype=torch.long)
    token_type_ids=torch.empty(len(examples),NUM_OPTIONS,max_seq_length,dtype=torch.long)
    labels=torch.empty(len(examples),dtype=torch.long)

    for example_index,example in enumerate(tqdm(examples)):
        #Process every option.
        for option_index,ending in enumerate(example.endings):
            encoding=encode_plus(mecab,tokenizer,example.question,ending,contexts[ending],max_length=max_seq_length)

            input_ids_tmp=encoding["input_ids"]
            token_type_ids_tmp=encoding["token_type_ids"]
            attention_mask_tmp=encoding["attention_mask"]

            input_ids[example_index,option_index]=input_ids_tmp
            token_type_ids[example_index,option_index]=token_type_ids_tmp
            attention_mask[example_index,option_index]=attention_mask_tmp

            if example_index==0 and option_index<4:
                logger.info("option_index={}".format(option_index))
                logger.info("question: {}".format(example.question))
                logger.info("ending: {}".format(ending))
                logger.info("context: {}".format(contexts[ending][:512]))
                logger.info("input_ids: {}".format(input_ids_tmp.detach().cpu().numpy()))
                logger.info("token_type_ids: {}".format(token_type_ids_tmp.detach().cpu().numpy()))
                logger.info("attention_mask: {}".format(attention_mask_tmp.detach().cpu().numpy()))

        labels[example_index]=example.label

    ret={
        "input_ids":input_ids,
        "attention_mask":attention_mask,
        "token_type_ids":token_type_ids,
        "labels":labels
    }

    return ret

def main(bert_model_dir,example_filepath,context_filepath,cache_save_dir):
    mecab=MeCab.Tagger("-d /usr/local/lib/mecab/dic/jumandic")

    logger.info("Cache files will be saved in {}.".format(cache_save_dir))

    #Tokenizer
    logger.info("Create a tokenizer from {}.".format(bert_model_dir))
    tokenizer = BertTokenizer.from_pretrained(bert_model_dir)

    logger.info("Start loading examples from {}.".format(example_filepath))
    examples=load_examples(example_filepath)
    logger.info("Finished loading examples.")
    logger.info("Number of examples: {}".format(len(examples)))

    logger.info("Start loading contexts from {}.".format(context_filepath))
    contexts=load_contexts(context_filepath)
    logger.info("Finished loading contexts.")

    logger.info("Start encoding examples.")
    encoding=encode_examples(mecab,tokenizer,examples,contexts,512)
    logger.info("Finished encoding examples.")

    os.makedirs(cache_save_dir,exist_ok=True)
    torch.save(encoding["input_ids"],os.path.join(cache_save_dir,"input_ids.pt"))
    torch.save(encoding["attention_mask"],os.path.join(cache_save_dir,"attention_mask.pt"))
    torch.save(encoding["token_type_ids"],os.path.join(cache_save_dir,"token_type_ids.pt"))
    torch.save(encoding["labels"],os.path.join(cache_save_dir,"labels.pt"))
    logger.info("Saved cache files in {}.".format(cache_save_dir))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="AIO")

    parser.add_argument("--bert_model_dir",type=str,default="~/BERTModels/NICT_BERT-base_JapaneseWikipedia_100K")
    parser.add_argument("--example_filepath",type=str,default="~/AIOData/train_questions.json")
    parser.add_argument("--context_filepath",type=str,default="~/AIOData/candidate_entities.json.gz")
    parser.add_argument("--cache_save_dir",type=str,default="~/EncodedCacheJumanpp/Train")

    args=parser.parse_args()

    main(args.bert_model_dir,args.example_filepath,args.context_filepath,args.cache_save_dir)
