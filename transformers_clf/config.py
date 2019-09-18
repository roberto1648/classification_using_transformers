## Required parameters
data_dir = "/path/to/train_and_dev_tsv/"
# help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
model_type = "bert"
# help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
model_name_or_path="bert-base-uncased"
# help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
task_name = 'mrpc'
# help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
output_dir = '/path/to/output/dir/'
# help="The output directory where the model predictions and checkpoints will be written.")
label_list = [0, 1]
# the processor is supposed to infer this, but it doesn't work, just put manually the labels

## Other parameters
config_name = ''
# help="Pretrained config name or path if not the same as model_name")
tokenizer_name = ''
# help="Pretrained tokenizer name or path if not the same as model_name")
cache_dir = ''
# help="Where do you want to store the pre-trained models downloaded from s3")
max_seq_length = 128
# help="The maximum total input sequence length after tokenization. Sequences longer "than this will be truncated, sequences shorter will be padded.")
do_train = True #, action='store_true',
# help="Whether to run training.")
do_eval = True #, action='store_true',
# help="Whether to run eval on the dev set.")
evaluate_during_training = True #, action='store_true',
# help="Rul evaluation during training at each logging step.")
do_lower_case = True #, action='store_true',
# help="Set this flag if you are using an uncased model.")

per_gpu_train_batch_size = 8
# help="Batch size per GPU/CPU for training.")
per_gpu_eval_batch_size = 8
# help="Batch size per GPU/CPU for evaluation.")
gradient_accumulation_steps = 1
# help="Number of updates steps to accumulate before performing a backward/update pass.")
learning_rate = 5e-5
# help="The initial learning rate for Adam.")
weight_decay = 0.0
# help="Weight decay if we apply some.")
adam_epsilon = 1e-8
# help="Epsilon for Adam optimizer.")
max_grad_norm = 1
# help="Max gradient norm.")
num_train_epochs = 3  # default = 3.0
# help="Total number of training epochs to perform.")
max_steps = -1
# help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
warmup_steps = 0
# help="Linear warmup over warmup_steps.")
max_in_training_eval_steps = 256
# help=Maximum number of validation samples to take when running on training loop

logging_steps = 50
# help="Log every X updates steps.")
save_steps = 24000 #50
# help="Save checkpoint every X updates steps.")
eval_all_checkpoints = False #True #", action='store_true',
# help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
no_cuda = False #True #", action='store_true',
# help="Avoid using CUDA when available")
overwrite_output_dir = True #', action='store_true',
# help="Overwrite the content of the output directory")
overwrite_cache = True #', action='store_true',
# help="Overwrite the cached training and evaluation sets")
seed = 42
# help="random seed for initialization")

fp16 = False #', action='store_true',
# help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
fp16_opt_level = '01' #, type=str, default='O1',
# help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].""See details at https://nvidia.github.io/apex/amp.html")
local_rank = -1
# help="For distributed training: local_rank")
server_ip = ''
# help="For distant debugging.")
server_port = ''
# help="For distant debugging.")

# n_gpu = 1

# def get_new_output_dir():
#     import os
#     parent_dir, __ = os.path.split(output_dir)
#     run_number = len([nm for nm in os.listdir(parent_dir)])
#     return os.path.join(parent_dir, f"run{run_number}")
