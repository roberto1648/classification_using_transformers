import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from .utils_glue import (convert_examples_to_features, output_modes, processors)

from . import config as args


def load_and_cache_examples(
        # required inputs:
        tokenizer,
        # inputs loaded by default from config.py:
        label_list=args.label_list,
        local_rank=args.local_rank, #-1,
        data_dir=args.data_dir, #"/mnt/StorageHD/data/imdb/tsv/",
        model_name_or_path=args.model_name_or_path, #"bert-base-uncased",
        max_seq_length=args.max_seq_length, #128,
        model_type=args.model_type, #'bert',
        task=args.task_name, #'mrpc',
        # label_list=args.label_list, #[0, 1],
        # other:
        # max_examples=-1,
        evaluate=False,
        logger=None,
):
    distributed_barrier(local_rank)
    processor = processors[task]()
    output_mode = output_modes[task]

    # Load data features from cache or dataset file
    cached_features_file = get_cache_file_path(
        data_dir, evaluate, model_name_or_path, max_seq_length, task
    )

    if os.path.exists(cached_features_file):# and (max_examples != -1):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        # label_list = [0, 1]  # processor.get_labels()
        examples = processor.get_dev_examples(data_dir)\
            if evaluate else processor.get_train_examples(data_dir)

        # # if max_examples is given take a random subset of the examples
        # rdn_idx_perm = np.random.permutation(range(len(examples)))[:max_examples]
        # examples = [examples[idx] for idx in rdn_idx_perm] if max_examples != -1 else examples
        # print("!!!!!!total number of examples:", len(examples))
        # # label_list = [0, 1]

        features = convert_examples_to_features(
            examples,
            label_list,
            max_seq_length,
            tokenizer,
            output_mode,
            cls_token_at_end = bool(model_type in ['xlnet']),
            # xlnet has a cls token at the end
            cls_token = tokenizer.cls_token,
            sep_token = tokenizer.sep_token,
            cls_token_segment_id =2 if model_type in ['xlnet'] else 0,
            pad_on_left = bool(model_type in ['xlnet']),
            # pad on the left for xlnet
            pad_token_segment_id = 4 if model_type in ['xlnet'] else 0
        )
        if local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    distributed_barrier(local_rank)
    dataset = convert_to_tensors_and_build_dataset(features, output_mode)

    return dataset


def get_cache_file_path(
        data_dir, evaluate, model_name_or_path, max_seq_length, task
):
    cached_features_file = os.path.join(
        data_dir,
        'cached_{}_{}_{}_{}'.format(
            'dev' if evaluate else 'train',
            list(filter(None, model_name_or_path.split('/'))).pop(),
            str(max_seq_length),
            str(task)
        )
    )
    return cached_features_file


def load_features_from_cache(cached_features_file, logger):
    logger.info("Loading features from cached file %s", cached_features_file)
    features = torch.load(cached_features_file)
    return features


def distributed_barrier(local_rank=-1):
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache


def convert_to_tensors_and_build_dataset(features, output_mode):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset