import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .utils_glue import compute_metrics
from .load_and_cache_examples import load_and_cache_examples
from . import config as args


def evaluate(
        # required inputs:
        tokenizer, 
        model,
        device,
        output_mode,
        n_gpu,
        label_list,
        # inputs loaded by default from config.py:
        output_dir=args.output_dir, #'/mnt/StorageHD/data/imdb/tsv/bert_runs/run3/',
        model_type=args.model_type, #'bert',
        task_name=args.task_name, #"mrpc",
        local_rank=args.local_rank, #-1,
        per_gpu_eval_batch_size=args.per_gpu_eval_batch_size, #8,
        # other:
        prefix="",
        max_examples=-1,
        logger=None,
):
    eval_task_names, eval_outputs_dirs = setup_double_evaluation_handling(
        task_name, output_dir,
    )
    results = {}

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, eval_batch_size, eval_dataloader = setup_task(
            tokenizer, label_list, eval_task, eval_output_dir, local_rank, n_gpu,
            per_gpu_eval_batch_size, max_examples,
            logger,
        )#;print("eval dataset lenght: ", len(eval_dataset))
        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            eval_loss, nb_eval_steps, preds, out_label_ids = eval_batch(
                    batch, model, device, model_type,
                    eval_loss, nb_eval_steps, preds, out_label_ids,
            )
            if manual_stop(): break

        eval_loss = eval_loss / nb_eval_steps
        result = get_metrics_from_preds(
            preds, output_mode, eval_task, out_label_ids
        )
        save_and_print_result(result, eval_output_dir, prefix, logger)
        results.update(result)
        if manual_stop(): break

    return results


def setup_double_evaluation_handling(task_name, output_dir):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if task_name == "mnli" \
        else (task_name,)
    eval_outputs_dirs = (output_dir, output_dir + '-MM') \
        if task_name == "mnli" else (output_dir,)
    return eval_task_names, eval_outputs_dirs


def setup_task(
        tokenizer, label_list, eval_task, eval_output_dir, local_rank, n_gpu,
        per_gpu_eval_batch_size, max_examples,
        logger,
):
    eval_dataset = load_and_cache_examples(
        tokenizer=tokenizer,
        label_list=label_list,
        task=eval_task,
        logger=logger,
        # max_examples=max_examples,
    )
    if not os.path.exists(eval_output_dir) and local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
    # Note that DistributedSampler samples randomly
    if max_examples != -1:
        eval_sampler = RandomSampler(eval_dataset, num_samples=max_examples, replacement=True) if local_rank == -1 \
            else DistributedSampler(eval_dataset)
    else:
        eval_sampler = SequentialSampler(eval_dataset) if local_rank == -1 \
            else DistributedSampler(eval_dataset)

    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=eval_batch_size
    )
    return eval_dataset, eval_batch_size, eval_dataloader


def eval_batch(
        batch, model, device, model_type,
        eval_loss, nb_eval_steps, preds, out_label_ids,
):
    model.eval()
    batch = tuple(t.to(device) for t in batch)

    with torch.no_grad():
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2] if model_type in ['bert', 'xlnet'] \
                else None,
            # XLM don't use segment_ids
            'labels': batch[3]}
        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]

        eval_loss += tmp_eval_loss.mean().item()

    nb_eval_steps += 1

    if preds is None:
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs['labels'].detach().cpu().numpy()
    else:
        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        out_label_ids = np.append(
            out_label_ids, inputs['labels'].detach().cpu().numpy(),
            axis=0
        )
    return eval_loss, nb_eval_steps, preds, out_label_ids


def get_metrics_from_preds(
      preds, output_mode, eval_task, out_label_ids
):
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)

    result = compute_metrics(eval_task, preds, out_label_ids)
    return result


def save_and_print_result(
        result, eval_output_dir, prefix, logger,
):
    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))

        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def manual_stop():
    stop = False

    if os.path.exists(".stop"):
        print("manual stop ('.stop' file detected)")
        stop = True

    return stop