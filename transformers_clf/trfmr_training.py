import random
import os
import numpy as np
from shutil import copyfile

# import config as args
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
import torch
from torch.utils.data.distributed import DistributedSampler

from .pytorch_transformers import AdamW, WarmupLinearSchedule
from tqdm import tqdm, trange

from .trfmr_evaluation import evaluate
from . import config as args


def train(
        # required inputs:
        train_dataset, 
        model, 
        tokenizer,
        device,
        output_mode,
        n_gpu,
        # inputs loaded by default from config.py:
        label_list=args.label_list,
        num_train_epochs=args.num_train_epochs,
        output_dir=args.output_dir,
        model_type=args.model_type,
        local_rank=args.local_rank,
        per_gpu_train_batch_size=args.per_gpu_train_batch_size,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        fp16_opt_level=args.fp16_opt_level,
        max_grad_norm=args.max_grad_norm,
        evaluate_during_training=args.evaluate_during_training,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        max_in_training_eval_steps=args.max_in_training_eval_steps,
        # other:
        logger=None,
):
    """ Train the model """
    delete_stop_file()
    tb_logdir = get_tb_logdir() if local_rank in [-1, 0] else None
    # tb_writer = SummaryWriter() if local_rank in [-1, 0] else None
    train_dataloader, train_batch_size = setup_dataloader(
         train_dataset, local_rank, n_gpu, per_gpu_train_batch_size,
    )
    t_total, num_train_epochs = get_total_steps_and_epochs(
        train_dataloader, max_steps,
        gradient_accumulation_steps, num_train_epochs,
    )
    optimizer, scheduler, model = prepare_optimizer_and_schedule(
            model, learning_rate, adam_epsilon, weight_decay, warmup_steps, t_total,
            fp16, fp16_opt_level,
    )
    model = setup_multi_gpu_training(model, n_gpu, local_rank)
    # Train!
    print_settings(
        len(train_dataset), num_train_epochs, per_gpu_train_batch_size,
        train_batch_size, gradient_accumulation_steps, local_rank, t_total, logger
    )
    # init training:
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(
        int(num_train_epochs),
        desc="Epoch",
        disable=local_rank not in [-1, 0]
    )
    set_seed(seed, n_gpu)  # Added here for reproductibility (even between python 2 and 3)

    for _ in train_iterator: # over epochs
        epoch_iterator = tqdm(
            train_dataloader,
            desc="Iteration",
            disable=local_rank not in [-1, 0]
        )

        for step, batch in enumerate(epoch_iterator): # over batches
            model.train()
            loss = loss_function(
                batch, model, device, model_type, n_gpu, gradient_accumulation_steps,
            )
            loss_backprop(
                loss, model.parameters(), optimizer, max_grad_norm, fp16,
            )
            tr_loss += loss.item() # scalar training loss for display/record
            global_step = optimizer_step(
                optimizer, scheduler, model,
                gradient_accumulation_steps, global_step, step,
            )
            curr_lr = scheduler.get_lr()[0] # current learning rate
            logging_loss = log_batch_metrics(
                tr_loss, logging_loss, curr_lr, tb_logdir, local_rank, logger,
                step, logging_steps, global_step, gradient_accumulation_steps,
                evaluate_during_training, tokenizer, model, device, output_mode,
                n_gpu, label_list, max_in_training_eval_steps,
            )
            save_model_checkpoint(
                model, step, gradient_accumulation_steps, global_step, save_steps,
                local_rank, output_dir, logger,
            )
            if (max_steps > 0 and global_step > max_steps) or manual_stop():
                epoch_iterator.close()
                break

        if (max_steps > 0 and global_step > max_steps) or manual_stop():
            train_iterator.close()
            break

    # not needed since we're opening/closing tb_writter inside the loop.
    # if local_rank in [-1, 0]:
    #     tb_writer.close()

    return global_step, tr_loss / global_step


def delete_stop_file():
    if os.path.exists(".stop"): os.remove(".stop")
    print("Starting training, to stop create a '.stop' file in cwd")


def manual_stop():
    stop = False

    if os.path.exists(".stop"):
        print("manual stop ('.stop' file detected)")
        stop = True

    return stop


def set_seed(seed=42, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0: torch.cuda.manual_seed_all(seed)


def setup_dataloader(
     train_dataset, local_rank, n_gpu, per_gpu_train_batch_size,
):
    train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
    train_sampler = RandomSampler(train_dataset) if local_rank == -1 \
        else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_batch_size
    )
    return train_dataloader, train_batch_size


def get_total_steps_and_epochs(
        train_dataloader, max_steps, gradient_accumulation_steps, num_train_epochs,
):
    if max_steps > 0:
        t_total = max_steps
        num_train_epochs = max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
    return t_total, num_train_epochs


def prepare_optimizer_and_schedule(
        model, learning_rate, adam_epsilon, weight_decay, warmup_steps, t_total,
        fp16, fp16_opt_level,
):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=adam_epsilon
    )
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        t_total=t_total
    )
    if fp16: # setup half resolution
        amp = try_to_import_apex()
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    return optimizer, scheduler, model


def try_to_import_apex():
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    return amp


def setup_multi_gpu_training(
        model, n_gpu, local_rank,
):
    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1: model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    return model


def print_settings(
       dset_length, num_train_epochs, per_gpu_train_batch_size, train_batch_size,
        gradient_accumulation_steps, local_rank, t_total, logger
):
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", dset_length) #len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                train_batch_size * gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


def loss_function(
        batch, model, device, model_type, n_gpu, gradient_accumulation_steps,
):
    batch = tuple(t.to(device) for t in batch)
    inputs = {
        'input_ids': batch[0],
        'attention_mask': batch[1],
        'token_type_ids': batch[2] if model_type in ['bert', 'xlnet'] else None,
        # XLM don't use segment_ids
        'labels': batch[3],
    }
    outputs = model(**inputs)
    loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

    if n_gpu > 1: loss = loss.mean()  # mean() to average on multi-gpu parallel training
    if gradient_accumulation_steps > 1: loss = loss / gradient_accumulation_steps

    return loss


def loss_backprop(
    loss, model_parameters, optimizer, max_grad_norm, fp16,
):
    if fp16:
        amp = try_to_import_apex()
        with amp.scale_loss(loss, optimizer) as scaled_loss: scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            amp.master_params(optimizer),
            max_grad_norm
        )
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_parameters, max_grad_norm)


def optimizer_step(
     optimizer, scheduler, model, gradient_accumulation_steps, global_step, step,
):
    if (step + 1) % gradient_accumulation_steps == 0:
        scheduler.step()  # Update learning rate schedule
        optimizer.step()
        model.zero_grad()
        global_step += 1

    return global_step


def log_batch_metrics(
    tr_loss, logging_loss, lr, tb_logdir, local_rank, logger,
    step, logging_steps, global_step, gradient_accumulation_steps,
    evaluate_during_training, tokenizer, model, device, output_mode,
    n_gpu, label_list, max_in_training_eval_steps,
):
    if (step + 1) % gradient_accumulation_steps == 0:
        if local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0:
            # Log metrics
            with SummaryWriter(tb_logdir) as tb_writer: # to avoid freezing tb on errors
                if local_rank == -1 and evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        output_mode=output_mode,
                        n_gpu=n_gpu,
                        label_list=label_list,
                        logger=logger,
                        max_examples=max_in_training_eval_steps,
                    )

                    for key, value in results.items():
                        tb_writer.add_scalar(
                            'eval_{}'.format(key),
                            value,
                            global_step
                        )

                tb_writer.add_scalar('lr', lr, global_step)
                tb_writer.add_scalar(
                    'loss',
                    (tr_loss - logging_loss) / logging_steps,
                    global_step
                )
                logging_loss = tr_loss
                # tb_writer.flush() # to save immediately (not needed since we're closing)

    return logging_loss


def save_model_checkpoint(
    model, step, gradient_accumulation_steps, global_step, save_steps,
    local_rank, output_dir, logger,
):
    if (step + 1) % gradient_accumulation_steps == 0:
        if local_rank in [-1, 0] and save_steps > 0 and global_step % save_steps == 0:
            # Save model checkpoint
            output_dir = os.path.join(output_dir, 'checkpoint-{}'.format(global_step))
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            # copy inputs in config.py
            copyfile(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py"),
                os.path.join(output_dir, 'training_py')
            )
            # torch.save(args, os.path.join(output_dir, 'training_bin'))
            logger.info("Saving model checkpoint to %s", output_dir)


def get_tb_logdir():
    with SummaryWriter() as tb_writter:
        logdir = tb_writter.logdir
    return logdir


# def train_on_batch(
#         batch, model, optimizer, scheduler, device, tokenizer,
#         model_type, n_gpu, local_rank,
#         global_step, tr_loss, logging_loss,
#         gradient_accumulation_steps, step, max_grad_norm,
#         logging_steps, save_steps, tb_writer, logger, evaluate_during_training,
#         output_mode, label_list, output_dir,
#         fp16,
# ):
#     model.train()
#     batch = tuple(t.to(device) for t in batch)
#     inputs = {'input_ids': batch[0],
#               'attention_mask': batch[1],
#               'token_type_ids': batch[2] if model_type in ['bert', 'xlnet'] else None,
#               # XLM don't use segment_ids
#               'labels': batch[3]}
#     outputs = model(**inputs)
#     loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
#
#     if n_gpu > 1:
#         loss = loss.mean()  # mean() to average on multi-gpu parallel training
#     if gradient_accumulation_steps > 1:
#         loss = loss / gradient_accumulation_steps
#
#     if fp16:
#         amp = try_to_import_apex()
#         with amp.scale_loss(loss, optimizer) as scaled_loss:
#             scaled_loss.backward()
#         torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
#     else:
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#
#     tr_loss += loss.item()
#     if (step + 1) % gradient_accumulation_steps == 0:
#         scheduler.step()  # Update learning rate schedule
#         optimizer.step()
#         model.zero_grad()
#         global_step += 1
#
#         if local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0:
#             # Log metrics
#             if local_rank == -1 and evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
#                 results = evaluate(
#                     tokenizer=tokenizer,
#                     model=model,
#                     device=device,
#                     output_mode=output_mode,
#                     n_gpu=n_gpu,
#                     label_list=label_list,
#                     logger=logger,
#                 )
#                 for key, value in results.items():
#                     tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
#             tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
#             tb_writer.add_scalar('loss', (tr_loss - logging_loss) / logging_steps, global_step)
#             logging_loss = tr_loss
#
#         if local_rank in [-1, 0] and save_steps > 0 and global_step % save_steps == 0:
#             # Save model checkpoint
#             output_dir = os.path.join(output_dir, 'checkpoint-{}'.format(global_step))
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#             model_to_save = model.module if hasattr(model,
#                                                     'module') else model  # Take care of distributed/parallel training
#             model_to_save.save_pretrained(output_dir)
#
#             copyfile(
#                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py"),
#                 os.path.join(output_dir, 'training_py')
#             )
#             # torch.save(args, os.path.join(output_dir, 'training_bin'))
#
#             logger.info("Saving model checkpoint to %s", output_dir)

