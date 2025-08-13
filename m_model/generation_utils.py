# -*- coding: utf-8 -*-
import torch
import copy
from typing import Optional, Callable, List, Union
import numpy as np
import torch.nn.functional as F
from transformers import (
    LogitsProcessorList, 
    HammingDiversityLogitsProcessor, 
    NoBadWordsLogitsProcessor, 
    MinLengthLogitsProcessor, 
    PrefixConstrainedLogitsProcessor, 
    ForcedBOSTokenLogitsProcessor, 
    ForcedEOSTokenLogitsProcessor, 
    InfNanRemoveLogitsProcessor, 
    RepetitionPenaltyLogitsProcessor, 
    NoRepeatNGramLogitsProcessor, 
    StoppingCriteriaList, 
    MaxLengthCriteria, 
    MaxTimeCriteria,
)
from m_model.beam_constraints import PhrasalConstraint
from m_model.beam_search import BeamSearchScorer, ConstrainedBeamSearchScorer
from m_utils.m_util import CLS, SEP, POS, NEG


def _get_logits_processor(
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    bad_words_ids: List[List[int]],
    min_length: int,
    max_length: int,
    eos_token_id: int,
    forced_bos_token_id: int,
    forced_eos_token_id: int,
    prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
    num_beams: int,
    num_beam_groups: int,
    diversity_penalty: float,
    remove_invalid_values: bool,
) -> LogitsProcessorList:
    """
    This mathod returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
    :obj:`~transformers.LogitsProcessor` instances used to modify the scores of the language model head.
    """
    processors = LogitsProcessorList()

    if diversity_penalty is not None and diversity_penalty > 0.0:
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
            )
        )
    if repetition_penalty is not None and repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if bad_words_ids is not None:
        processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
    if min_length is not None and eos_token_id is not None and min_length > -1:
        processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
    if prefix_allowed_tokens_fn is not None:
        processors.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams // num_beam_groups))
    if forced_bos_token_id is not None:
        processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
    if forced_eos_token_id is not None:
        processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
    if remove_invalid_values is True:
        processors.append(InfNanRemoveLogitsProcessor())
    return processors

def _get_stopping_criteria(max_length: Optional[int], max_time: Optional[float]) -> StoppingCriteriaList:
    stopping_criteria = StoppingCriteriaList()
    if max_length is not None:
        stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    if max_time is not None:
        stopping_criteria.append(MaxTimeCriteria(max_time=max_time))
    return stopping_criteria
    
def _validate_stopping_criteria(stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
    stopping_max_length = stopping_criteria.max_length
    new_stopping_criteria = copy.deepcopy(stopping_criteria)
    if stopping_max_length is not None and stopping_max_length != max_length:
        print ("You set different `max_length` for stopping criteria and `max_length` parameter", flush=True)
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    return new_stopping_criteria

def greedy_decoding(
    model,
    inputs,
    tokenizer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    phrase_generation: Optional[bool] = None,
    question_generation: Optional[bool] = None,
    compute_phr_loss: Optional[bool] = None,
    compute_ques_loss: Optional[bool] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    ) -> torch.LongTensor:
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        stopping_criteria = _validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    scores = () if (return_dict_in_generate and output_scores) else None

    if phrase_generation and not question_generation:
        input_ids = inputs["batch_conans"][0]
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        input_ids, input_masks = inputs["batch_conans"]    
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        context_ids, context_mask = inputs['batch_context']
        answer_ids, answer_mask = inputs['batch_answer']
        dec_ids = torch.tensor([tokenizer.bos_token_id, tokenizer.convert_tokens_to_ids(POS)]).unsqueeze(0).repeat(batch_size, 1).contiguous().to(device)
        while True:
            model_out = model(
                input_ids=input_ids, attention_mask=input_masks, context_ids=context_ids, context_mask=context_mask,
                answer_ids=answer_ids, answer_mask=answer_mask, decoder_input_ids=None, decoder_attention_mask=None,
                decoder_phrase_input_ids=dec_ids, decoder_phrase_mask=None, keyphrase_label=None, keyphrase_label_mask=None,
                question_label=None, question_label_mask=None, phrase_generation=phrase_generation, question_generation=question_generation,
                compute_phr_loss=compute_phr_loss, compute_ques_loss=compute_ques_loss
            )
            # process logits
            next_token_logits = model_out["phr_logits"][:, -1, :]
            next_tokens_scores = logits_processor(dec_ids, next_token_logits)
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            dec_ids = torch.cat([dec_ids, next_tokens[:, None]], dim=-1)
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if unfinished_sequences.max() == 0 or stopping_criteria(dec_ids, scores):
                break
        
    elif question_generation and not phrase_generation:
        input_ids = inputs["batch_conans"][0]
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        input_ids, input_masks = inputs["batch_conans"]    
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        context_ids, context_mask = inputs['batch_context']
        answer_ids, answer_mask = inputs['batch_answer']
        keyphrase_ids, keyphrase_mask = inputs['batch_keyphrase']
        tf_keyphrase_ids, tf_keyphrase_mask = inputs['tf_keyphrase']
        dec_ids = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).repeat(batch_size, 1).contiguous().to(device)
        while True:
            model_out = model(
                input_ids=input_ids, attention_mask=input_masks, context_ids=context_ids, context_mask=context_mask,
                answer_ids=answer_ids, answer_mask=answer_mask, decoder_input_ids=dec_ids, decoder_attention_mask=None,
                decoder_phrase_input_ids=keyphrase_ids, decoder_phrase_mask=keyphrase_mask, keyphrase_label=None, keyphrase_label_mask=None,
                question_label=None, question_label_mask=None, phrase_generation=phrase_generation, question_generation=question_generation,
                compute_phr_loss=compute_phr_loss, compute_ques_loss=compute_ques_loss, tf_keyphrase_ids = tf_keyphrase_ids, tf_keyphrase_mask = tf_keyphrase_mask
            )
            # process logits
            next_token_logits = model_out["ques_logits"][:, -1, :]
            next_tokens_scores = logits_processor(dec_ids, next_token_logits)
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            dec_ids = torch.cat([dec_ids, next_tokens[:, None]], dim=-1)
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if unfinished_sequences.max() == 0 or stopping_criteria(dec_ids, scores):
                break
        
    return dec_ids

def model_decode(model, inputs, tokenizer, args):
    if args.mode == "test_phrase":
        max_length = args.max_phrase_len
    elif args.mode == "test_question":
        max_length = args.max_question_len
    else:
        max_length = 0
    print("**************************")
    print("max_length -> ", max_length)
    print("args.mode -> ", args.mode)
    print("**************************")
    
    assert max_length > 0
    top_k = args.top_k or 1
    top_p = args.top_p or 1.0
    phrase_generation = args.phrase_generation
    question_generation = args.question_generation
    compute_phr_loss = args.compute_phr_loss
    compute_ques_loss = args.compute_ques_loss

    min_length = args.min_length or 0
    repetition_penalty = args.repetition_penalty or None
    diversity_penalty = args.diversity_penalty or None
    no_repeat_ngram_size = args.no_repeat_ngram_size or None
    bad_words_ids = args.bad_words_ids or None
    remove_invalid_values = args.remove_invalid_values or False

    # get logits processor
    logits_processor = _get_logits_processor(
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        bad_words_ids=bad_words_ids,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=tokenizer.eos_token_id,
        forced_bos_token_id=tokenizer.bos_token_id,
        forced_eos_token_id=tokenizer.eos_token_id,
        prefix_allowed_tokens_fn=None,
        num_beams=1,
        num_beam_groups=1,
        diversity_penalty=diversity_penalty,
        remove_invalid_values=remove_invalid_values,
    )
    # get decoding stopping criteria
    stopping_criteria = _get_stopping_criteria(max_length=max_length, max_time=None)
    
    # apply decoding
    output = greedy_decoding(
        model=model,
        inputs=inputs,
        tokenizer=tokenizer,
        logits_processor=logits_processor,
        stopping_criteria=stopping_criteria,
        max_length=max_length,
        phrase_generation=phrase_generation,
        question_generation=question_generation,
        compute_phr_loss=compute_phr_loss,
        compute_ques_loss=compute_ques_loss
    )

    return output


def model_decode_beam(model, inputs, tokenizer, args, num_beams=8,
                      output_scores: bool = False):
    if args.mode == "test_phrase":
        max_length = args.max_phrase_len
    elif args.mode == "test_question":
        max_length = args.max_question_len
    else:
        max_length = 0
    print("**************************")
    print("beam search -> ", num_beams)
    print("max_length -> ", max_length)
    print("args.mode -> ", args.mode)
    print("**************************")
    
    assert max_length > 0
    phrase_generation = args.phrase_generation
    question_generation = args.question_generation
    compute_phr_loss = args.compute_phr_loss
    compute_ques_loss = args.compute_ques_loss

    min_length = args.min_length or 0
    repetition_penalty = args.repetition_penalty or None
    diversity_penalty = args.diversity_penalty or None
    no_repeat_ngram_size = args.no_repeat_ngram_size or None
    bad_words_ids = args.bad_words_ids or None
    remove_invalid_values = args.remove_invalid_values or False
    
    logits_processor = _get_logits_processor(
        diversity_penalty=diversity_penalty,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=tokenizer.eos_token_id,
        forced_bos_token_id=tokenizer.bos_token_id,
        forced_eos_token_id=tokenizer.eos_token_id,
        prefix_allowed_tokens_fn=None,
        num_beams=num_beams,
        num_beam_groups=1,
        bad_words_ids=bad_words_ids,
        remove_invalid_values=remove_invalid_values
    )
    
    stopping_criteria = _get_stopping_criteria(
        max_length=max_length, max_time=None)

    if phrase_generation and not question_generation:
        bsz_ctx, _ = inputs["batch_context"]
        
        beam_scorer = BeamSearchScorer(
            batch_size=bsz_ctx.size(0),
            num_beams=num_beams,
            device=bsz_ctx.device
        )
        batch_size = bsz_ctx.size(0)
        device = bsz_ctx.device
        dec_ids = torch.tensor([tokenizer.bos_token_id, tokenizer.convert_tokens_to_ids(POS)]).unsqueeze(0).repeat(batch_size, 1).contiguous().to(device)
        inputs['batch_keyphrase'][0] = dec_ids
        model_inputs = _expand_inputs_for_keyphrase(inputs, expand_size=num_beams)
        keyphrase_ids, _ = model_inputs['batch_keyphrase']
        
        batch_beam_size = keyphrase_ids.size(0)
        assert len(beam_scorer._beam_hyps) == batch_size
        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        scores = None
        beam_indices = None
        if output_scores:
            scores = ()
        
        conans_ids, conans_mask = model_inputs['batch_conans']
        context_ids, context_mask = model_inputs['batch_context']
        answer_ids, answer_mask = model_inputs['batch_answer']
            
        while True:
            model_outputs = model(
                input_ids=conans_ids, attention_mask=conans_mask, context_ids=context_ids, context_mask=context_mask,
                answer_ids=answer_ids, answer_mask=answer_mask, decoder_input_ids=None, decoder_attention_mask=None,
                decoder_phrase_input_ids=keyphrase_ids, decoder_phrase_mask=None, keyphrase_label=None, keyphrase_label_mask=None,
                question_label=None, question_label_mask=None, phrase_generation=phrase_generation, question_generation=question_generation,
                compute_phr_loss=compute_phr_loss, compute_ques_loss=compute_ques_loss
            )
            next_logits = model_outputs["phr_logits"][:, -1, :]
            next_scores = F.log_softmax(next_logits, dim=-1)
            next_scores_processed = logits_processor(keyphrase_ids, next_scores)
            next_scores = next_scores_processed + beam_scores[:, None].expand_as(next_scores)
            # reshape for beam search
            vocab_size = next_scores.shape[-1]

            next_scores = next_scores.view(
                batch_size, num_beams * vocab_size)
            next_scores, next_tokens = torch.topk(
                next_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            next_indices = (next_tokens // vocab_size).long()
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                keyphrase_ids,
                next_scores,
                next_tokens,
                next_indices,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                beam_indices=beam_indices
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            keyphrase_ids = torch.cat(
                [keyphrase_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            if beam_scorer.is_done or stopping_criteria(keyphrase_ids, scores):
                break

        seq_outputs = beam_scorer.finalize(
            keyphrase_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices
        )
        best_seqs = seq_outputs["sequences"]

    elif not phrase_generation and question_generation:
        bsz_ctx, _ = inputs["batch_context"]
        
        beam_scorer = BeamSearchScorer(
            batch_size=bsz_ctx.size(0),
            num_beams=num_beams,
            device=bsz_ctx.device
        )
        batch_size = bsz_ctx.size(0)
        device = bsz_ctx.device
        
        # keyphrase decoding
        dec_ids = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).repeat(batch_size, 1).contiguous().to(device)
        inputs['batch_question_decoder'][0] = dec_ids
        model_inputs = _expand_inputs_for_question(inputs, expand_size=num_beams)
        question_decoder_ids, _ = model_inputs['batch_question_decoder']
        
        batch_beam_size = question_decoder_ids.size(0)
        assert len(beam_scorer._beam_hyps) == batch_size
        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        scores = None
        beam_indices = None
        if output_scores:
            scores = ()
        
        conans_ids, conans_mask = model_inputs['batch_conans']
        context_ids, context_mask = model_inputs['batch_context']
        answer_ids, answer_mask = model_inputs['batch_answer']
        tf_keyphrase_ids, tf_keyphrase_mask = model_inputs['tf_keyphrase']
            
        while True:
            model_outputs = model(
                input_ids=conans_ids, attention_mask=conans_mask, context_ids=context_ids, context_mask=context_mask,
                answer_ids=answer_ids, answer_mask=answer_mask, decoder_input_ids=question_decoder_ids, decoder_attention_mask=None,
                decoder_phrase_input_ids=None, decoder_phrase_mask=None, keyphrase_label=None, keyphrase_label_mask=None,
                question_label=None, question_label_mask=None, phrase_generation=phrase_generation, question_generation=question_generation,
                compute_phr_loss=compute_phr_loss, compute_ques_loss=compute_ques_loss, tf_keyphrase_ids=tf_keyphrase_ids, tf_keyphrase_mask=tf_keyphrase_mask,
            )
            next_logits = model_outputs["ques_logits"][:, -1, :]
            next_scores = F.log_softmax(next_logits, dim=-1)
            next_scores_processed = logits_processor(question_decoder_ids, next_scores)
            next_scores = next_scores_processed + beam_scores[:, None].expand_as(next_scores)
            # reshape for beam search
            vocab_size = next_scores.shape[-1]

            next_scores = next_scores.view(
                batch_size, num_beams * vocab_size)
            next_scores, next_tokens = torch.topk(
                next_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            next_indices = (next_tokens // vocab_size).long()
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                question_decoder_ids,
                next_scores,
                next_tokens,
                next_indices,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                beam_indices=beam_indices
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            question_decoder_ids = torch.cat(
                [question_decoder_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            if beam_scorer.is_done or stopping_criteria(question_decoder_ids, scores):
                break

        seq_outputs = beam_scorer.finalize(
            question_decoder_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices
        )
        best_seqs = seq_outputs["sequences"]
        
    return best_seqs

def _expand_inputs_for_keyphrase(inputs, expand_size=1):
    bsz_cas_ids, bsz_cas_mask = inputs['batch_conans']
    bsz_ctx_ids, bsz_ctx_mask = inputs['batch_context']
    bsz_ans_ids, bsz_ans_mask = inputs['batch_answer']
    
    kph_input_ids, _ = inputs['batch_keyphrase']
    expanded_return_idx = (
        torch.arange(kph_input_ids.shape[0]).view(
            -1, 1).repeat(1, expand_size).view(-1).to(kph_input_ids.device)
    )
    
    kph_input_ids = kph_input_ids.index_select(0, expanded_return_idx)
    
    bsz_cas_ids = bsz_cas_ids.index_select(0, expanded_return_idx)
    bsz_cas_mask = bsz_cas_mask.index_select(0, expanded_return_idx)
    bsz_ctx_ids = bsz_ctx_ids.index_select(0, expanded_return_idx)
    bsz_ctx_mask = bsz_ctx_mask.index_select(0, expanded_return_idx)
    bsz_ans_ids = bsz_ans_ids.index_select(0, expanded_return_idx)
    bsz_ans_mask = bsz_ans_mask.index_select(0, expanded_return_idx)
    
    inputs['batch_conans'][0] = bsz_cas_ids
    inputs['batch_conans'][1] = bsz_cas_mask
    inputs['batch_context'][0] = bsz_ctx_ids
    inputs['batch_context'][1] = bsz_ctx_mask
    inputs['batch_answer'][0] = bsz_ans_ids
    inputs['batch_answer'][1] = bsz_ans_mask
    
    # kph_input_mask = kph_input_mask.index_select(0, expanded_return_idx)
    inputs['batch_keyphrase'][0] = kph_input_ids
    # inputs['batch_keyphrase'][1] = kph_input_mask
    
    return inputs
    
def _expand_inputs_for_question(inputs, expand_size=1):
    bsz_cas_ids, bsz_cas_mask = inputs['batch_conans']
    bsz_ctx_ids, bsz_ctx_mask = inputs['batch_context']
    bsz_ans_ids, bsz_ans_mask = inputs['batch_answer']
    bsz_tf_ids, bsz_tf_mask = inputs['tf_keyphrase']
    qes_input_ids, _ = inputs['batch_question_decoder']
    expanded_return_idx = (
        torch.arange(qes_input_ids.shape[0]).view(
            -1, 1).repeat(1, expand_size).view(-1).to(qes_input_ids.device)
    )
    bsz_cas_ids = bsz_cas_ids.index_select(0, expanded_return_idx)
    bsz_cas_mask = bsz_cas_mask.index_select(0, expanded_return_idx)
    bsz_ctx_ids = bsz_ctx_ids.index_select(0, expanded_return_idx)
    bsz_ctx_mask = bsz_ctx_mask.index_select(0, expanded_return_idx)
    bsz_ans_ids = bsz_ans_ids.index_select(0, expanded_return_idx)
    bsz_ans_mask = bsz_ans_mask.index_select(0, expanded_return_idx)
    bsz_tf_ids = bsz_tf_ids.index_select(0, expanded_return_idx)
    bsz_tf_mask = bsz_tf_mask.index_select(0, expanded_return_idx)
    inputs['batch_conans'][0] = bsz_cas_ids
    inputs['batch_conans'][1] = bsz_cas_mask
    inputs['batch_context'][0] = bsz_ctx_ids
    inputs['batch_context'][1] = bsz_ctx_mask
    inputs['batch_answer'][0] = bsz_ans_ids
    inputs['batch_answer'][1] = bsz_ans_mask
    inputs['tf_keyphrase'][0] = bsz_tf_ids
    inputs['tf_keyphrase'][1] = bsz_tf_mask
    qes_input_ids = qes_input_ids.index_select(0, expanded_return_idx)
    # qes_input_mask = qes_input_mask.index_select(0, expanded_return_idx)
    inputs['batch_question_decoder'][0] = qes_input_ids
    # inputs['batch_question_decoder'][1] = qes_input_mask
    return inputs