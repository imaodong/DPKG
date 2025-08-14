# -*- coding: utf-8 -*-
import logging
import os
import numpy as np
import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
from transformers.optimization import get_linear_schedule_with_warmup, AdamW
from tqdm import tqdm

class m_trainer():
    def __init__(self, model, train_loader, dev_loader, args, tokenizer) -> None:
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.tokenizer = tokenizer
        
        self.log_dir = args.log_dir
        self.log_steps = args.log_steps
        self.validate_steps = args.validate_steps
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.warmup_ratio = args.warmup_ratio
        self.weight_decay = args.weight_decay
        self.max_grad_norm = args.max_grad_norm
        
        self.phrase_generation = args.phrase_generation
        self.question_generation = args.question_generation
        self.compute_phr_loss = args.compute_phr_loss
        self.compute_ques_loss = args.compute_ques_loss

        total_steps = len(train_loader) * self.num_epochs
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
            num_warmup_steps=self.warmup_ratio * total_steps, 
            num_training_steps=total_steps)
        self.best_metric = 0.0

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)


    def train(self):
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info("Total parameters: {}\tTrainable parameters: {}".format(total_num, trainable_num))

        save_dir = os.path.join(self.log_dir,"best_params")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        best_model_store_path = os.path.join(save_dir,"best_model.bin")

        logging.info("Total batches per epoch : {}".format(len(self.train_loader)))
        logging.info("Evaluate every {} batches.".format(self.validate_steps))
        for epoch in range(self.num_epochs):
            logging.info("\nEpoch {}:".format(epoch + 1))
            for batch_step, inputs in enumerate(tqdm(self.train_loader)):
                self.model.train()
                conans_ids, conans_mask = inputs['batch_conans']
                context_ids, context_mask = inputs['batch_context']
                answer_ids, answer_mask = inputs['batch_answer']
                keyphrase_ids, keyphrase_mask = inputs['batch_keyphrase']
                question_decoder_ids, question_decoder_mask = inputs['batch_question_decoder']
                keyphrase_labels, keyphrase_labels_mask = inputs['batch_keyphrase_labels']
                question_labels, question_labels_mask = inputs['batch_question_labels']
                tf_keyphrase_ids, tf_keyphrase_mask = inputs['tf_keyphrase']
                
                model_output = self.model(input_ids = conans_ids, attention_mask = conans_mask, context_ids = context_ids, context_mask = context_mask, 
                                          answer_ids = answer_ids, answer_mask = answer_mask, decoder_input_ids = question_decoder_ids, decoder_attention_mask = question_decoder_mask,
                                          decoder_phrase_input_ids = keyphrase_ids, decoder_phrase_mask = keyphrase_mask, keyphrase_label = keyphrase_labels, 
                                          keyphrase_label_mask = keyphrase_labels_mask, question_label = question_labels, question_label_mask = question_labels_mask,
                                          phrase_generation = self.phrase_generation, question_generation = self.question_generation, compute_phr_loss = self.compute_phr_loss, compute_ques_loss = self.compute_ques_loss,
                                          tf_keyphrase_ids = tf_keyphrase_ids, tf_keyphrase_mask = tf_keyphrase_mask)
                
                phr_loss = model_output['phr_loss']
                ques_loss = model_output['ques_loss']
                phr_acc = model_output['phr_acc']
                phr_tot_tokens = model_output['phr_tot_tokens']
                ques_acc = model_output['ques_acc']
                ques_tot_tokens = model_output['ques_tot_tokens']
                
                avg_pool_loss = model_output['avg_pool_loss']
                loss = avg_pool_loss
                loss.backward()
                if self.max_grad_norm > 0:
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                if batch_step > 0 and batch_step % self.log_steps == 0:
                    logging.info("Batch Step: {}\tphrase_acc: {:.3f}\tphrase_loss: {:.5f}".format(batch_step, phr_acc / phr_tot_tokens, phr_loss.item()))
                    logging.info("Batch Step: {}\tquestion_acc: {:.3f}\tquestion_loss: {:.5f}".format(batch_step, ques_acc / ques_tot_tokens, ques_loss.item()))
                    logging.info("Batch Step: {}\tavg_pooling: {:.5f}".format(batch_step, avg_pool_loss.item()))
                    logging.info("Batch Step: {}\ttrain_loss: {:.5f}".format(batch_step, loss.item()))
                    weights = model_output['normalized_weights']
                    logging.info("Batch Step: {}\tweight1: {:.6f}, weight2: {:.6f}".format(batch_step, weights[0].item(), weights[1].item()))

                # validation
                if batch_step > 0 and batch_step % self.validate_steps == 0:
                    logging.info("Evaluating...")
                    predicts_dict = self.evaluate(loader = self.dev_loader)
                    logging.info("Evaluation phr_acc: {:.3f} phr_loss: {:.5f}, ques_acc: {:.3f} ques_loss: {:.5f}, avg_loss: {:.5f}".format(
                        predicts_dict["phr_avg_acc"], predicts_dict["phr_avg_loss"], predicts_dict["ques_avg_acc"], predicts_dict["ques_avg_loss"], predicts_dict['avg_pooling_loss'])
                    )
                    metric_acc = predicts_dict["phr_avg_acc"]
                    
                    if metric_acc > self.best_metric:
                        self.best_metric = metric_acc
                        logging.info("Epoch {} Batch Step {} -- Best phr_acc: {:.3f} phr_loss: {:.5f}, --Best ques_acc: {:.3f} ques_loss: {:.5f}, --avg_pooing_loss: {:.5f}, --Best acc: {:.3f}".
                                     format(epoch, batch_step, predicts_dict['phr_avg_acc'], predicts_dict['phr_avg_loss'], predicts_dict['ques_avg_acc'],
                                            predicts_dict['ques_avg_loss'], predicts_dict['avg_pooling_loss'], self.best_metric))
                        torch.save(self.model, best_model_store_path)
                        logging.info("Saved to [%s]" % best_model_store_path)
                        
            predicts_dict = self.evaluate(loader = self.dev_loader)
            metric_acc = predicts_dict["phr_avg_acc"]
            
            if metric_acc > self.best_metric:
                self.best_metric = metric_acc
                logging.info("Epoch {} -- Best phr_acc: {:.3f} phr_loss: {:.5f}, --Best ques_acc: {:.3f}, ques_loss: {:.5f}, --avg_pooing_loss: {:.5f}, --Best acc: {:.3f}".
                             format(epoch, predicts_dict['phr_avg_acc'], predicts_dict['phr_avg_loss'], predicts_dict['ques_avg_acc'],
                                    predicts_dict['ques_avg_loss'], predicts_dict['avg_pooling_loss'], self.best_metric))
                torch.save(self.model, best_model_store_path)
                logging.info("Saved to [%s]" % best_model_store_path)
                
            logging.info("Epoch {} training done.".format(epoch))
            model_to_save = os.path.join(self.log_dir, "model_epoch_%d.bin"%(epoch))
            torch.save(self.model, model_to_save)
            logging.info("Saved to [%s]" % model_to_save)
            
    def evaluate(self, loader):
        self.model.eval()
        
        phr_total_acc = 0.0
        phr_count_tok = 0.0
        phr_loss_list = []
        ques_total_acc = 0.0
        ques_count_tok = 0.0
        ques_loss_list = []
        avg_pooling_list = []
        for inputs in tqdm(loader):
            with torch.no_grad():
                conans_ids, conans_mask = inputs['batch_conans']
                context_ids, context_mask = inputs['batch_context']
                answer_ids, answer_mask = inputs['batch_answer']
                keyphrase_ids, keyphrase_mask = inputs['batch_keyphrase']
                question_decoder_ids, question_decoder_mask = inputs['batch_question_decoder']
                keyphrase_labels, keyphrase_labels_mask = inputs['batch_keyphrase_labels']
                question_labels, question_labels_mask = inputs['batch_question_labels']
                tf_keyphrase_ids, tf_keyphrase_mask = inputs['tf_keyphrase']
                
                model_output = self.model(input_ids = conans_ids, attention_mask = conans_mask, context_ids = context_ids, context_mask = context_mask, 
                                          answer_ids = answer_ids, answer_mask = answer_mask, decoder_input_ids = question_decoder_ids, decoder_attention_mask = question_decoder_mask,
                                          decoder_phrase_input_ids = keyphrase_ids, decoder_phrase_mask = keyphrase_mask, keyphrase_label = keyphrase_labels, 
                                          keyphrase_label_mask = keyphrase_labels_mask, question_label = question_labels, question_label_mask = question_labels_mask,
                                          phrase_generation = self.phrase_generation, question_generation = self.question_generation, compute_phr_loss = self.compute_phr_loss, compute_ques_loss = self.compute_ques_loss,
                                          tf_keyphrase_ids = tf_keyphrase_ids, tf_keyphrase_mask = tf_keyphrase_mask)
                phr_loss = model_output['phr_loss']
                ques_loss = model_output['ques_loss']
                pool_loss = model_output['avg_pool_loss']
                
                phr_acc = model_output['phr_acc']
                phr_tot_tokens = model_output['phr_tot_tokens']
                ques_acc = model_output['ques_acc']
                ques_tot_tokens = model_output['ques_tot_tokens']
                
                phr_total_acc += phr_acc
                phr_count_tok += phr_tot_tokens
                phr_loss_list.append(float(phr_loss))
                ques_total_acc += ques_acc
                ques_count_tok += ques_tot_tokens
                ques_loss_list.append(float(ques_loss))
                avg_pooling_list.append(float(pool_loss))
        phr_avg_acc = phr_total_acc / phr_count_tok
        phr_avg_loss = np.mean(phr_loss_list)
        ques_avg_acc = ques_total_acc / ques_count_tok
        ques_avg_loss = np.mean(ques_loss_list)
        avg_pooling_loss = np.mean(avg_pooling_list)
        
        return_dict = {
            "phr_avg_acc": phr_avg_acc,
            "phr_avg_loss": phr_avg_loss,
            "ques_avg_acc": ques_avg_acc,
            "ques_avg_loss": ques_avg_loss,
            "avg_pooling_loss": avg_pooling_loss
        }
        return return_dict   



