# -*- coding: utf-8 -*-
import torch

def max_seq_length(list_l):
    return max(len(l) for l in list_l)

def pad_sequence(list_l, max_len, padding_value = 0):
    assert len(list_l) <= max_len
    padding_l = [padding_value] * (max_len - len(list_l))
    padded_list = list_l + padding_l
    return padded_list


class My_collator(object):

    def __init__(self, device, padding_idx, args):
        self.device = torch.device('cuda')
        self.padding_idx = padding_idx
        self.phrase_generation = args.phrase_generation
        self.question_generation = args.question_generation
        self.compute_phr_loss = args.compute_phr_loss
        self.compute_ques_loss = args.compute_ques_loss
    
    def list_to_tensor(self, list_l):
        max_len = max_seq_length(list_l)
        padded_lists = []
        for list_seq in list_l:
            padded_lists.append(pad_sequence(list_seq, max_len, padding_value = self.padding_idx))
        input_tensor = torch.tensor(padded_lists, dtype = torch.long)
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor

    def list_to_tensor_v2(self, list_l):
        max_len = max_seq_length(list_l)
        padded_lists = []
        for list_seq in list_l:
            padded_lists.append(pad_sequence(list_seq, max_len, padding_value = -100))
        input_tensor = torch.tensor(padded_lists, dtype = torch.long)
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor

    def varlist_to_tensor(self, list_vl):
        lens = []
        for list_l in list_vl:
            lens.append(max_seq_length(list_l))
        max_len = max(lens)
        
        padded_lists = []
        for list_seqs in list_vl:
            v_list = []
            for list_l in list_seqs:
                v_list.append(pad_sequence(list_l, max_len, padding_value=self.padding_idx))
            padded_lists.append(v_list)
        input_tensor = torch.tensor(padded_lists, dtype=torch.long)
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor
    
    def get_attention_mask(self, data_tensor: torch.tensor):
        attention_mask = torch.zeros_like(data_tensor)
        attention_mask = attention_mask.masked_fill(data_tensor == self.padding_idx, 0)
        attention_mask = attention_mask.masked_fill(data_tensor != self.padding_idx, 1)
        attention_mask = attention_mask.to(self.device).contiguous()
        return attention_mask
    
    
    def get_attention_mask_v2(self, data_tensor: torch.tensor):
        attention_mask = torch.zeros_like(data_tensor)
        attention_mask = attention_mask.masked_fill(data_tensor == -100, 0)
        attention_mask = attention_mask.masked_fill(data_tensor != -100, 1)
        attention_mask = attention_mask.to(self.device).contiguous()
        return attention_mask


    def custom_collate(self, mini_batch):
        conans_ids_list = []
        context_ids_list = []
        answer_ids_list = []
        keyphrase_input_ids_list = []
        keyphrase_labels_list = []
        question_decoder_input_ids_list = []
        question_labels_list = []
        tf_keyphrase_list = []

        for sample in mini_batch:
            conans_ids_list.append(sample.conans_ids)
            context_ids_list.append(sample.context_ids)
            answer_ids_list.append(sample.answer_ids)
            keyphrase_input_ids_list.append(sample.keyphrase_input_ids)
            keyphrase_labels_list.append(sample.keyphrase_labels)
            question_decoder_input_ids_list.append(sample.question_decoder_input_ids)
            question_labels_list.append(sample.question_labels)
            tf_keyphrase_list.append(sample.tf_keyphrase)

        if self.phrase_generation and self.question_generation and self.compute_phr_loss and self.compute_ques_loss:
            batch_conans_ids = self.list_to_tensor(conans_ids_list)
            batch_conans_mask = self.get_attention_mask(batch_conans_ids)

            batch_context_ids = self.list_to_tensor(context_ids_list)
            batch_context_mask = self.get_attention_mask(batch_context_ids)
            
            batch_answer_ids = self.list_to_tensor(answer_ids_list)
            batch_answer_mask = self.get_attention_mask(batch_answer_ids)
            
            batch_keyphrase_ids = self.list_to_tensor(keyphrase_input_ids_list)
            batch_keyphrase_mask = self.get_attention_mask(batch_keyphrase_ids)

            batch_keyphrase_labels = self.list_to_tensor_v2(keyphrase_labels_list)
            batch_keyphrase_labels_mask = self.get_attention_mask_v2(batch_keyphrase_labels)
            
            batch_question_decoder_ids = self.list_to_tensor(question_decoder_input_ids_list)
            batch_question_decoder_mask = self.get_attention_mask(batch_question_decoder_ids)

            batch_question_labels = self.list_to_tensor_v2(question_labels_list)
            batch_question_labels_mask = self.get_attention_mask_v2(batch_question_labels)
        
            batch_tf_keyphrase = self.list_to_tensor(tf_keyphrase_list)
            batch_tf_keyphrase_mask = self.get_attention_mask(batch_tf_keyphrase)
        
        
        elif self.phrase_generation and not self.question_generation and not self.compute_phr_loss and not self.compute_ques_loss:
            batch_conans_ids = self.list_to_tensor(conans_ids_list)
            batch_conans_mask = self.get_attention_mask(batch_conans_ids)

            batch_context_ids = self.list_to_tensor(context_ids_list)
            batch_context_mask = self.get_attention_mask(batch_context_ids)
            
            batch_answer_ids = self.list_to_tensor(answer_ids_list)
            batch_answer_mask = self.get_attention_mask(batch_answer_ids)
            
            batch_keyphrase_ids = None
            batch_keyphrase_mask = None
            batch_question_decoder_ids = None
            batch_question_decoder_mask = None
            batch_keyphrase_labels = None
            batch_keyphrase_labels_mask = None
            batch_question_labels = None
            batch_question_labels_mask = None
            batch_tf_keyphrase = None
            batch_tf_keyphrase_mask = None

        elif not self.phrase_generation and self.question_generation and not self.compute_phr_loss and not self.compute_ques_loss:
            batch_conans_ids = self.list_to_tensor(conans_ids_list)
            batch_conans_mask = self.get_attention_mask(batch_conans_ids)

            batch_context_ids = self.list_to_tensor(context_ids_list)
            batch_context_mask = self.get_attention_mask(batch_context_ids)
            
            batch_answer_ids = self.list_to_tensor(answer_ids_list)
            batch_answer_mask = self.get_attention_mask(batch_answer_ids)
            
            batch_keyphrase_ids = None
            batch_keyphrase_mask = None

            batch_question_decoder_ids = None
            batch_question_decoder_mask = None
            batch_keyphrase_labels = None
            batch_keyphrase_labels_mask = None
            batch_question_labels = None
            batch_question_labels_mask = None
            
            batch_tf_keyphrase = self.list_to_tensor(tf_keyphrase_list)
            batch_tf_keyphrase_mask = self.get_attention_mask(batch_tf_keyphrase)
            
        collated_batch = {
            "batch_conans": [batch_conans_ids, batch_conans_mask], 
            "batch_context": [batch_context_ids, batch_context_mask],
            "batch_answer": [batch_answer_ids, batch_answer_mask],
            "batch_keyphrase": [batch_keyphrase_ids, batch_keyphrase_mask],
            
            "batch_question_decoder": [batch_question_decoder_ids, batch_question_decoder_mask],
            "batch_keyphrase_labels": [batch_keyphrase_labels, batch_keyphrase_labels_mask],
            "batch_question_labels": [batch_question_labels, batch_question_labels_mask],
            "tf_keyphrase": [batch_tf_keyphrase, batch_tf_keyphrase_mask]
        }

        return collated_batch


