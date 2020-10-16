import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from seq2seq_trainer import Seq2SeqTrainer, arg_to_scheduler_choices
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import EvaluationStrategy
import numpy as np
from transformers import BartTokenizer, T5Tokenizer
import torch
from transformers.modeling_bart import shift_tokens_right

def lmap(f, x):
    """list(map(f, x))"""
    return list(map(f, x))

@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Parameters:
        label_smoothing (:obj:`float`, `optional`, defaults to 0):
            The label smoothing epsilon to apply (if not zero).
        sortish_sampler (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to SortishSamler or not. It sorts the inputs according to lenghts in-order to minimizing the padding size.
        predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    """
    def __init__(self):
        self.label_smoothing = 0.0 
        self.sortish_sampler = False 
        self.predict_with_generate = False
        self.adafactor = False
        self.encoder_layerdrop = None
        self.decoder_layerdrop = None
        self.dropout = None
        self.attention_dropout = None
        self.lr_scheduler = "linear"
        self.output_dir = "./result"
        self.logging_dir = "./logging"
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    def __init__(self):
        self.data_dir = ""
        self.task = "summarization"
        self.max_source_length = 1024
        self.max_target_length  = 128
        self.val_max_target_length  = 142
        self.test_max_target_length = 142
        self.n_train  = -1
        self.n_val  = -1
        self.n_test  = -1
        self.src_lang  = None
        self.tgt_lang  = None
        self.eval_beams  = None
def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class Seq2SeqDataCollator:
    def __init__(self, tokenizer, data_args, tpu_num_cores=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores
        self.dataset_kwargs = {"add_prefix_space": isinstance(tokenizer, BartTokenizer)}
        if data_args.src_lang is not None:
            self.dataset_kwargs["src_lang"] = data_args.src_lang
        if data_args.tgt_lang is not None:
            self.dataset_kwargs["tgt_lang"] = data_args.tgt_lang

    def __call__(self, batch):
        if hasattr(self.tokenizer, "prepare_seq2seq_batch"):
            batch = self._encode(batch)
            input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )
        else:
            input_ids = torch.stack([x["input_ids"] for x in batch])
            attention_mask = torch.stack([x["attention_mask"] for x in batch])
            labels = torch.stack([x["labels"] for x in batch])

            labels = trim_batch(labels, self.pad_token_id)
            input_ids, attention_mask = trim_batch(input_ids, self.pad_token_id, attention_mask=attention_mask)

        if isinstance(self.tokenizer, T5Tokenizer):
            decoder_input_ids = self._shift_right_t5(labels)
        else:
            decoder_input_ids = shift_tokens_right(labels, self.pad_token_id)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
        return batch

    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.pad_token_id
        return shifted_input_ids

    def _encode(self, batch):
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.data_args.max_source_length,
            max_target_length=self.data_args.max_target_length,
            padding="max_length" if self.tpu_num_cores is not None else "longest",  # TPU hack
            return_tensors="pt",
            **self.dataset_kwargs,
        )
        return batch_encoding.data
