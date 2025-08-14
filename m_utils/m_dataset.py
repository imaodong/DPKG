# -*- coding: utf-8 -*-
import logging
import os
import pickle
import dataclasses
import json
from dataclasses import dataclass
from typing import List
from torch.utils.data import Dataset
from tqdm import tqdm
from m_utils.m_util import CLS, SEP, POS, NEG


@dataclass(frozen=True)
class InputFeature:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    conans_ids: List[int]
    context_ids: List[int]
    answer_ids: List[int]
    keyphrase_input_ids: List[int]
    keyphrase_labels: List[int]
    question_decoder_input_ids: List[int]
    question_labels: List[int]
    tf_keyphrase: List[int]

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class My_dataset():
    def __init__(self,
                 data_path,
                 tokenizer,
                 data_partition,
                 cache_dir,
                 max_context_len,
                 max_answer_len,
                 is_test = False,
                 is_question = False) -> None:
        self.tokenizer = tokenizer
        self.data_partition = data_partition
        self.cache_dir = cache_dir
        self.max_context_len = max_context_len
        self.max_answer_len = max_answer_len
        self.max_conans_len = max_context_len + max_answer_len
        self.is_test = is_test
        self.is_question = is_question
        self.instances = []
        self._cache_instances(data_path)

    def _cache_instances(self,data_path):
        if self.is_question:
            signature = "{}_ques_cache.pkl".format(self.data_partition)
        else:
            signature = "{}_cache.pkl".format(self.data_partition)
        
        if self.cache_dir is not None:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            cache_path = os.path.join(self.cache_dir, signature)
        else:
            cache_dir = os.mkdir("caches")
            cache_path = os.path.join(cache_dir, signature)
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                logging.info("Loading cached instances from {}".format(cache_path))
                self.instances = pickle.load(f)
        else:
            logging.info("Loading raw data from {}".format(data_path))
            with open(data_path, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            all_samples = []
            for sample in tqdm(data):
                context_ = sample['context']
                support_fact_ = sample['supporting_facts']
                cont = ""
                for sub_sf in support_fact_:
                    sf_str, sf_idx = sub_sf[0], sub_sf[1]
                    for sub_context in context_:
                        if sub_context[0] == sf_str:
                            sub_cont = ""
                            for s1 in sub_context[1]:
                                sub_cont = sub_cont + s1
                            cont = cont + sub_cont + " "
                
                all_samples.append({
                    "context": cont,
                    "answer": sample['answer'],
                    "p_phrase": sample['p_phrase'],
                    "n_phrase": sample['n_phrase'],
                    "question": sample['question']
                })

            logging.info("Creating cache instances {}".format(signature))
            if self.is_question:
                kp_ids, kp_labels, tf_kp_keyphrase = self._parse_keyphrase_ids(keyphrase_file='./outputs/best_model_test.jsonl')
                assert len(kp_ids) == len(kp_labels) == len(all_samples) == len(tf_kp_keyphrase)
                
            cc = 0
            for row in tqdm(all_samples):
                conans_ids, context_ids, answer_ids, keyphrase_input_ids, keyphrase_labels, question_decoder_input_ids, question_labels, tf_keyphrase = self._parse_input_ids(row['context'],row['answer'],row['p_phrase'],row['n_phrase'],row['question'])
                inputs = {
                    'conans_ids': conans_ids,
                    'context_ids': context_ids,
                    'answer_ids': answer_ids,
                    'keyphrase_input_ids': keyphrase_input_ids,
                    'keyphrase_labels': keyphrase_labels,
                    'question_decoder_input_ids': question_decoder_input_ids,
                    'question_labels': question_labels,
                    'tf_keyphrase': tf_keyphrase
                }
                
                if self.is_question:
                    inputs['keyphrase_input_ids'] = kp_ids[cc]
                    inputs['keyphrase_labels'] = kp_labels[cc]
                    inputs['tf_keyphrase'] = tf_kp_keyphrase[cc]
                    feature = InputFeature(**inputs)
                    self.instances.append(feature)
                else:
                    feature = InputFeature(**inputs)
                    self.instances.append(feature)
                cc += 1    
            logging.info("The number of sample: {}".format(cc))
                
            with open(cache_path,'wb') as f:
                pickle.dump(self.instances, f)


    def _parse_input_ids(self, context: str, answer: str, p_phrase: list, n_phrase: list, question: str):

        context = "Context: " + context
        context_tokens = self.tokenizer.tokenize(context)
        context_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
        if len(context_ids) > self.max_context_len - 2:
            context_ids = context_ids[:self.max_context_len - 2]
        context_ids = self.tokenizer.convert_tokens_to_ids([CLS]) + context_ids + self.tokenizer.convert_tokens_to_ids([SEP])

        answer = "Answer: " + answer
        answer_tokens = self.tokenizer.tokenize(answer)
        answer_ids = self.tokenizer.convert_tokens_to_ids(answer_tokens)
        answer_ids = self.tokenizer.convert_tokens_to_ids([CLS]) + answer_ids + self.tokenizer.convert_tokens_to_ids([SEP])

        conans_ids = context_ids + answer_ids[1:]

        question = "Question: " + question
        question_tokens = self.tokenizer.tokenize(question)
        question_ids = self.tokenizer.convert_tokens_to_ids(question_tokens)
        question_decoder_input_ids = self.tokenizer.convert_tokens_to_ids([CLS]) + question_ids
        question_labels = question_ids + self.tokenizer.convert_tokens_to_ids([SEP])
        
        if self.is_test:
            keyphrase_input = []
        else:
            keyphrase_input = []
            for pp in p_phrase:  
                keyphrase_input += [POS]
                keyphrase_input += [pp]

            for np in n_phrase:
                keyphrase_input += [NEG]
                keyphrase_input += [np]
                
        keyphrase_string = "".join(keyphrase_input) 
        keyphrase_tokens = self.tokenizer.tokenize(keyphrase_string)
        keyphrase_input_ids = self.tokenizer.convert_tokens_to_ids([CLS] + keyphrase_tokens)
        keyphrase_labels = self.tokenizer.convert_tokens_to_ids(keyphrase_tokens + [SEP])
        tf_keyphrase = self.tokenizer.convert_tokens_to_ids([CLS] + keyphrase_tokens + [SEP])
      
        return conans_ids, context_ids, answer_ids, keyphrase_input_ids, keyphrase_labels, question_decoder_input_ids, question_labels, tf_keyphrase

    def _parse_keyphrase_ids(self, keyphrase_file):
        keyphrase_datas = []
        with open(keyphrase_file, 'r', encoding='utf-8') as fr:
            for le in tqdm(fr):
                se = json.loads(le.strip())
                keyphrase_datas.append(se['keyphrase'])
        key_ids = []
        key_labels = []
        tf_key_phrases = []
        
        for kd in keyphrase_datas:
            k_tokens = self.tokenizer.tokenize(kd)
            k_ids = self.tokenizer.convert_tokens_to_ids([CLS] + k_tokens)
            key_ids.append(k_ids)
            k_i_labels = self.tokenizer.convert_tokens_to_ids(k_tokens + [SEP])
            key_labels.append(k_i_labels)
            tf_key_phrase_ids =  self.tokenizer.convert_tokens_to_ids([CLS] + k_tokens + [SEP])
            tf_key_phrases.append(tf_key_phrase_ids)
            
        return key_ids, key_labels, tf_key_phrases

    def __len__(self):
        return len(self.instances)

    def __getitem__(self,index):
        return self.instances[index]


    



