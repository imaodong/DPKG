# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import argparse
import sys
import logging
import json
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from m_utils.m_util import get_tokenizer, combine_tokens
from m_utils.m_dataset import My_dataset
from m_utils.m_collator import My_collator
from m_model.modeling_bart import MyQGModel
from m_utils.m_trainer import m_trainer
from transformers import BartConfig



logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
  handlers=[
      logging.StreamHandler(sys.stdout)
  ]
)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type = int, default = 8)
    parser.add_argument('--max_context_len', type = int, default = 405)
    parser.add_argument('--max_question_len', type = int, default = 60)
    parser.add_argument('--max_phrase_len', type = int, default = 130)
    parser.add_argument('--max_answer_len', type = int, default = 0)
    parser.add_argument('--cache_dir', type = str, default = "caches")
    parser.add_argument('--bart_dir', type = str, default = '/data/models/mdli/Bart/bart-base/')
    parser.add_argument('--phrase_generation', type = bool, default = True)
    parser.add_argument('--question_generation', type = bool, default = True)
    parser.add_argument('--compute_phr_loss', type = bool, default = True)
    parser.add_argument('--compute_ques_loss', type = bool, default = True)

    parser.add_argument('--log_dir', type = str, default = "logs")
    parser.add_argument('--train_file', type = str, default = 'm_data/HotpotQA/train.json')
    parser.add_argument('--dev_file', type = str, default = 'm_data/HotpotQA/dev.json')
    parser.add_argument('--test_file', type = str, default = 'm_data/HotpotQA/test.json')
    parser.add_argument('--gpus', type = bool, default = True) # 多gpu
    parser.add_argument('--embed_dim', type = int, default = 768)
    parser.add_argument('--random_seed', type = int, default = 42)
    parser.add_argument('--log_steps', type = int, default = 100) 
    parser.add_argument('--validate_steps', type = int, default = 2000)

    # Training
    parser.add_argument('--mode', type = str, default = None)
    parser.add_argument('--lr', type = float, default = 1e-5)
    parser.add_argument('--weight_decay', type = float, default = 0.01)
    parser.add_argument('--warmup_ratio', type = float, default = 0.1)
    parser.add_argument('--max_grad_norm', type = float, default = 1.0)
    parser.add_argument('--num_epochs', type = int, default = 50)

    # Generate
    parser.add_argument('--infer_checkpoint', type = str, default = None)
    parser.add_argument('--output_dir', type = str, default = "outputs")
    parser.add_argument('--decoding_mode', type = str, default = "greedy", choices = ["greedy", "topk", "topp"])
    # parser.add_argument('--max_length', type = int, default = 80)
    parser.add_argument('--repetition_penalty', type = float, default = 1.0)
    parser.add_argument('--no_repeat_ngram_size', type = int, default = 0)
    parser.add_argument('--bad_words_ids', type = list, default = None)
    parser.add_argument('--min_length', type = int, default = 0)
    parser.add_argument('--diversity_penalty', type = float, default = 0.0)
    parser.add_argument('--output_scores', type = bool, default = False)
    parser.add_argument('--return_dict_in_generate', type = bool, default = False)
    parser.add_argument('--remove_invalid_values', type = bool, default = False)
    parser.add_argument('--top_k', type = int, default = 10)
    parser.add_argument('--top_p', type = float, default = 0.9)
    
    return parser.parse_args()

def set_seed(args):
    if args.random_seed is not None:
        random.seed(args.random_seed) # 
        np.random.seed(args.random_seed) # 
        torch.cuda.manual_seed_all(args.random_seed) # 
        torch.cuda.manual_seed(args.random_seed) # 
        torch.backends.cudnn.deterministic = True # 
        torch.backends.cudnn.benchmark = False # 
        torch.manual_seed(args.random_seed) # 
        torch.random.manual_seed(args.random_seed)
   

def print_args(args):
    print("=============== Args ===============")
    for k in vars(args):
        print("%s: %s" % (k, vars(args)[k]))

def run_train(args):
    logging.info("=============== Training ===============")
    device = torch.device("cuda")
    args.device = device
    tokenizer, num_added_tokens, token_id_dict = get_tokenizer(config_dir = args.bart_dir)
    args.pad_token_id = token_id_dict["pad_token_id"]

    train_dataset = My_dataset(data_path = args.train_file, tokenizer = tokenizer, data_partition = 'train',\
        cache_dir = args.cache_dir, max_context_len = args.max_context_len, max_answer_len = args.max_answer_len)
    dev_dataset = My_dataset(data_path = args.dev_file, tokenizer = tokenizer, data_partition = 'dev',\
        cache_dir = args.cache_dir, max_context_len = args.max_context_len, max_answer_len = args.max_answer_len)

    collator = My_collator(device=device, padding_idx = args.pad_token_id, args=args)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, collate_fn = collator.custom_collate)
    dev_loader = DataLoader(dev_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = collator.custom_collate)
    
    device_ids = [0,1,2,3]
    
    config = BartConfig.from_pretrained(args.bart_dir)
    config.use_cache = False
    qg_model = MyQGModel.from_pretrained(args.bart_dir, config = config, args = args)
    qg_model.resize_token_embeddings(len(tokenizer))
    # qg_model = torch.load('./logs/best_params/best_model.bin')
    qg_model.cuda()
    # qg_model = nn.DataParallel(qg_model,device_ids=device_ids).cuda()

    Ttrainer = m_trainer(qg_model, train_loader, dev_loader, args, tokenizer)
    Ttrainer.train()

def run_test_phrase(args):
    logging.info("===============Phrase Testing ===============")
    args.phrase_generation = True
    args.question_generation = False
    args.compute_phr_loss = False
    args.compute_ques_loss = False
    device = torch.device("cuda")
    tokenizer, num_added_tokens, token_id_dict = get_tokenizer(config_dir=args.bart_dir)
    args.pad_token_id = token_id_dict["pad_token_id"]
    test_dataset = My_dataset(data_path = args.test_file, tokenizer = tokenizer, data_partition = 'test', cache_dir = args.cache_dir, max_context_len = args.max_context_len, max_answer_len = args.max_answer_len, is_test = True)
    test_collator = My_collator(device = device, padding_idx = args.pad_token_id, args=args)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = test_collator.custom_collate)
    
    if args.infer_checkpoint is not None:
        model_path = os.path.join(args.log_dir, args.infer_checkpoint)
    else:
        model_path = os.path.join(args.log_dir, "best_params/best_model.bin")
    model = torch.load(model_path)
    logging.info("Model loaded from [{}]".format(model_path))
    model = model.to(device)
    model.eval()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_prefix = model_path.split('/')[-1].replace('.bin','_test.jsonl')
    output_path = os.path.join(args.output_dir, output_prefix)
    
    with open(output_path, 'w', encoding = 'utf-8') as ff:
        for idx, inputs in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                output = model.generate(inputs=inputs, tokenizer=tokenizer, args=args)
                sentences = combine_tokens(output, tokenizer, mod = "test_phrase")
            for k_phrase in sentences:
                keyphrases = {"keyphrase": k_phrase}
                line = json.dumps(keyphrases, ensure_ascii = False)
                ff.write(line + '\n')
                ff.flush()
    logging.info("Saved output to [{}]".format(output_path))


def run_test_question(args):
    logging.info("=============== Question Testing ===============")
    device = torch.device("cuda")
    args.phrase_generation = False
    args.question_generation = True
    args.compute_phr_loss = False
    args.compute_ques_loss = False
    tokenizer, num_added_tokens, token_id_dict = get_tokenizer(config_dir=args.bart_dir)
    args.pad_token_id = token_id_dict["pad_token_id"]
    test_dataset = My_dataset(data_path = args.test_file, tokenizer = tokenizer, data_partition = 'test', cache_dir = args.cache_dir, max_context_len = args.max_context_len, max_answer_len = args.max_answer_len, is_test = True, is_question = True)
    test_collator = My_collator(device = device, padding_idx = args.pad_token_id, args = args)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = test_collator.custom_collate)
    
    if args.infer_checkpoint is not None:
        model_path = os.path.join(args.log_dir, args.infer_checkpoint)
    else:
        model_path = os.path.join(args.log_dir, "best_params/best_model.bin")
    model = torch.load(model_path)
    logging.info("Model loaded from [{}]".format(model_path))
    model.to(device)
    model.eval()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_prefix = model_path.split('/')[-1].replace('.bin','_question.jsonl')
    output_path = os.path.join(args.output_dir, output_prefix)
    with open(output_path, 'w', encoding = 'utf-8') as ff:
        for idx, inputs in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                output = model.generate(inputs=inputs, tokenizer=tokenizer, args=args)
                sentences = combine_tokens(output, tokenizer, mod = "test_question")
            for question in sentences:
                questions = {"question": question}
                line = json.dumps(questions, ensure_ascii = False)
                ff.write(line + '\n')
                ff.flush()
    logging.info("Saved output to [{}]".format(output_path))


if __name__ == "__main__":
    args = parse_config()
    set_seed(args)
    
    if args.mode == "train":
        print_args(args)
        run_train(args)
    elif args.mode == "test_phrase":
        run_test_phrase(args)
    elif args.mode == "test_question":
        run_test_question(args)
    
    else:
        exit("Please specify the \"mode\" parameter!")




