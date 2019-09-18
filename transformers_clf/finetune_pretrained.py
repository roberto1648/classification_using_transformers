# modified examples/run_glue.py
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
from shutil import copyfile
import numpy as np
import torch
# from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
#                               TensorDataset)
# from torch.utils.data.distributed import DistributedSampler
# from tensorboardX import SummaryWriter
# from tqdm import tqdm, trange

from .pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

# from .pytorch_transformers import AdamW, WarmupLinearSchedule

from .utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)
# import sys  # to run argparse from notebook
# import config as args # all parameter configuration settings stored at config.py
from .trfmr_training import train as trfmr_train
from .trfmr_evaluation import evaluate as trfmr_evaluate
from .load_and_cache_examples import load_and_cache_examples
from . import config as args

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig)),
                 ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
}


def main():
    # args = parse_args()
    # from . import config as args;print("starting")
    # args = get_args_object()
    delete_stop_file()
    check_output_dir(
        args.output_dir, args.overwrite_output_dir, args.do_train
    )
    setup_distant_debugging_if_needed(
        args.server_ip, args.server_port
    )
    device, n_gpu = setup_CUDA_GPU_and_distributed_training(
        args.local_rank, args.no_cuda
    )
    setup_logging(
        args.local_rank, device, n_gpu, args.fp16
    )
    set_seed(
        args.seed, n_gpu
    )
    processor, label_list, num_labels, output_mode =\
        prepare_GLUE_task(args.task_name)

    model, tokenizer, model_class, tokenizer_class =\
        load_pretrained_model_and_tokenizer(
            device=device,
            num_labels=num_labels
        )

    if args.do_train:
        train(model, tokenizer, device, output_mode, n_gpu, args.label_list)

    model, tokenizer = save_reload_model_and_tokenizer(
        model, tokenizer, model_class, tokenizer_class, device,
    )

    if args.do_eval and args.local_rank in [-1, 0]:
        evaluate(tokenizer, model_class, device, output_mode, n_gpu, args.label_list)


# def get_args_object():
#     from . import config
#     names = [nm for nm in dir(config) if not nm.startswith("__")]
#     class args(): pass
#     for nm in names: setattr(args, nm, getattr(config, nm))
#     return args


def check_output_dir(output_dir="", overwrite_output_dir=True, do_train=True):
    if os.path.exists(output_dir) and os.listdir(
            output_dir) and do_train and not overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                output_dir))


def setup_distant_debugging_if_needed(server_ip='', server_port=''):
    if server_ip and server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(server_ip, server_port), redirect_output=True)
        ptvsd.wait_for_attach()


def setup_CUDA_GPU_and_distributed_training(local_rank=-1, no_cuda=False):
    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = 1
    return device, n_gpu


def setup_logging(local_rank=-1, device=torch.device('cpu'), n_gpu=1, fp16=False):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if local_rank in [-1, 0] else logging.WARN
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        local_rank, device, n_gpu, bool(local_rank != -1),
        fp16
    )


def prepare_GLUE_task(task_name='mrpc'):
    task_name = task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    return processor, label_list, num_labels, output_mode


def load_pretrained_model_and_tokenizer(
        local_rank=args.local_rank, #-1,
        model_type=args.model_type, #"bert",
        config_name=args.config_name, #'',
        model_name_or_path=args.model_name_or_path, #"bert-base-uncased",
        task_name=args.task_name, #'mrpc',
        tokenizer_name=args.tokenizer_name, #'',
        do_lower_case=args.do_lower_case, #True,
        device=torch.device("cuda"),
        num_labels=1,
):
    # Make sure only the first process in distributed training will download model & vocab
    if local_rank not in [-1, 0]: torch.distributed.barrier()

    model_type = model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(
        config_name if config_name else model_name_or_path,
        num_labels=num_labels, finetuning_task=task_name
    )
    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path,
        do_lower_case=do_lower_case
    )
    model = model_class.from_pretrained(
        model_name_or_path, from_tf=bool('.ckpt' in model_name_or_path),
        config=config
    )

    # Make sure only the first process in distributed training will download model & vocab
    if local_rank == 0: torch.distributed.barrier()

    model.to(device)

    # logger.info("Training/evaluation parameters %s", args)

    return model, tokenizer, model_class, tokenizer_class


def  train(model, tokenizer, device, output_mode, n_gpu, label_list):
    train_dataset = load_and_cache_examples(
        tokenizer=tokenizer,
        label_list=label_list,
        evaluate=False,
        logger=logger,
    ) # other default kwargs from config.py
    global_step, tr_loss = trfmr_train(
        train_dataset=train_dataset,
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_mode=output_mode,
        n_gpu=n_gpu,
        label_list=label_list,
        logger=logger,
    ) # other default kwargs from config.py
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


def save_reload_model_and_tokenizer(
        model, tokenizer, model_class, tokenizer_class, device,
        do_train=args.do_train,
        local_rank=args.local_rank,
        output_dir=args.output_dir,
):
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if do_train and (local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(output_dir) and local_rank in [-1, 0]:
            os.makedirs(output_dir)

        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        copyfile(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py"),
            os.path.join(output_dir, 'training_args.py')
        )
        # torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(output_dir)
        tokenizer = tokenizer_class.from_pretrained(output_dir)
        model.to(device)
        return model, tokenizer


def evaluate(
        tokenizer, model_class, device, output_mode, n_gpu, label_list,
        eval_all_checkpoints=args.eval_all_checkpoints,
        output_dir=args.output_dir,
):
    results = {}
    checkpoints = [output_dir]

    if eval_all_checkpoints:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(
                glob.glob(output_dir + '/**/' + WEIGHTS_NAME, recursive=True)
            )
        )
        logging.getLogger(
            "pytorch_transformers.modeling_utils"
        ).setLevel(logging.WARN)  # Reduce logging

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
        if manual_stop(): break
        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
        model = model_class.from_pretrained(checkpoint)
        model.to(device)
        result = trfmr_evaluate(
            tokenizer=tokenizer,
            model=model,
            device=device,
            output_mode=output_mode,
            n_gpu=n_gpu,
            label_list=label_list,
            prefix=global_step,
            logger=logger
        ) # other default kwargs from config.py
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)


def set_seed(seed=42, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0: torch.cuda.manual_seed_all(seed)


def delete_stop_file():
    if os.path.exists(".stop"): os.remove(".stop")
    print("Starting training, to stop create a '.stop' file in cwd")


def manual_stop():
    stop = False

    if os.path.exists(".stop"):
        print("manual stop ('.stop' file detected)")
        stop = True

    return stop


if __name__ == "__main__":
    main()
