import argparse
import gzip
import json
import logging
import os
import sys
from tqdm import tqdm
import torch
from transformers import BertJapaneseTokenizer
import jaconv

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

def parse_with_mecab(mecab,text):
    mecab.parse("")
    node=mecab.parseToNode(text)

    tokens=[]
    while node:
        word=node.surface
        tokens.append(word)

        node=node.next

    return tokens

def encode_examples(tokenizer,examples,contexts,max_seq_length):
    input_ids=torch.empty(len(examples),NUM_OPTIONS,max_seq_length,dtype=torch.long)
    attention_mask=torch.empty(len(examples),NUM_OPTIONS,max_seq_length,dtype=torch.long)
    token_type_ids=torch.empty(len(examples),NUM_OPTIONS,max_seq_length,dtype=torch.long)
    labels=torch.empty(len(examples),dtype=torch.long)

    for example_index,example in enumerate(tqdm(examples)):
        #Process every option.
        for option_index,ending in enumerate(example.endings):
            str_question=jaconv.h2z(example.question,kana=True,digit=True,ascii=True)
            str_ending=jaconv.h2z(ending,kana=True,digit=True,ascii=True)
            str_context=jaconv.h2z(contexts[ending],kana=True,digit=True,ascii=True)

            #Text features
            text_a=str_question+"[SEP]"+str_ending
            text_b=str_context

            encoding = tokenizer.encode_plus(
                text_a,
                text_b,
                return_tensors="pt",
                add_special_tokens=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                max_length=max_seq_length,
                truncation=True,
                truncation_strategy="only_second"   #Truncate the context.
            )

            input_ids_tmp=encoding["input_ids"].view(-1)
            token_type_ids_tmp=encoding["token_type_ids"].view(-1)
            attention_mask_tmp=encoding["attention_mask"].view(-1)

            input_ids[example_index,option_index]=input_ids_tmp
            token_type_ids[example_index,option_index]=token_type_ids_tmp
            attention_mask[example_index,option_index]=attention_mask_tmp

            if example_index==0 and option_index<4:
                logger.info("option_index={}".format(option_index))
                logger.info("text_a: {}".format(text_a[:512]))
                logger.info("text_b: {}".format(text_b[:512]))
                logger.info("input_ids: {}".format(input_ids_tmp.detach().cpu().numpy()))
                logger.info("token_type_ids: {}".format(token_type_ids_tmp.detach().cpu().numpy()))
                logger.info("attention_mask: {}".format(attention_mask_tmp.detach().cpu().numpy()))

        labels[example_index]=example.label

    return input_ids,attention_mask,token_type_ids,labels

def main(bert_vocab_filepath,example_filepath,context_filepath,cache_save_dir):
    logger.info("Cache files will be saved in {}.".format(cache_save_dir))

    #Tokenizer
    logger.info("Create a tokenizer from {}.".format(bert_vocab_filepath))
    tokenizer = BertJapaneseTokenizer.from_pretrained(bert_vocab_filepath)

    logger.info("Start loading examples from {}.".format(example_filepath))
    examples=load_examples(example_filepath)
    logger.info("Finished loading examples.")
    logger.info("Number of examples: {}".format(len(examples)))

    logger.info("Start loading contexts from {}.".format(context_filepath))
    contexts=load_contexts(context_filepath)
    logger.info("Finished loading contexts.")

    logger.info("Start encoding examples.")
    input_ids,attention_mask,token_type_ids,labels=encode_examples(tokenizer,examples,contexts,512)
    logger.info("Finished encoding examples.")

    os.makedirs(cache_save_dir,exist_ok=True)
    torch.save(input_ids,os.path.join(cache_save_dir,"input_ids.pt"))
    torch.save(attention_mask,os.path.join(cache_save_dir,"attention_mask.pt"))
    torch.save(token_type_ids,os.path.join(cache_save_dir,"token_type_ids.pt"))
    torch.save(labels,os.path.join(cache_save_dir,"labels.pt"))
    logger.info("Saved cache files in {}.".format(cache_save_dir))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="AIO")

    parser.add_argument("--bert_vocab_filepath",type=str,default="~/BERTModels/NICT_BERT-base_JapaneseWikipedia_100K/vocab.txt")
    parser.add_argument("--example_filepath",type=str,default="~/AIOData/train_questions.json")
    parser.add_argument("--context_filepath",type=str,default="~/AIOData/candidate_entities.json.gz")
    parser.add_argument("--cache_save_dir",type=str,default="~/EncodedCache/Train")

    args=parser.parse_args()

    main(args.bert_vocab_filepath,args.example_filepath,args.context_filepath,args.cache_save_dir)
