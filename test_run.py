import argparse
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from trainer import Trainer
from utils import init_logger, load_tokenizer, read_prediction_text, MODEL_CLASSES, MODEL_PATH_MAP, set_seed
import logging
from data_loader import load_and_cache_examples
from main import get_args_parser

logger = logging.getLogger(__name__)


args = get_args_parser()
init_logger()                       # 输出信息
set_seed(args)                      # 设置随机种子
tokenizer = load_tokenizer(args)    # 加载预训练模型
train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
dev_dataset   = load_and_cache_examples(args, tokenizer, mode="dev")
test_dataset  = load_and_cache_examples(args, tokenizer, mode="test")

# trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

train_sampler = RandomSampler(train_dataset)
print(train_sampler)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
print(train_dataloader)

slot_label_lst = get_slot_labels(args)


