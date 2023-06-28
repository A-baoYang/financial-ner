import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from model import JointBERT
from opencc import OpenCC
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    BertTokenizer,
)

root_dir = Path(__name__).parent.absolute()
convertor_zhtw_zhcn = OpenCC("tw2sp")
MODEL_CLASSES = {
    "bert": (BertConfig, JointBERT, BertTokenizer),
    "chinese_bert_wwm": (BertConfig, JointBERT, BertTokenizer),
    "chatglm": (AutoConfig, JointBERT, AutoTokenizer),
}

MODEL_PATH_MAP = {
    # 'bert':'./data/bert-base-chinese'
    "bert": "bert-base-chinese",
    "chinese_bert_wwm": "hfl/chinese-roberta-wwm-ext-large",
    "chatglm": "THUDM/chatglm-6b",
}
PRETRAINED_MODEL_MAP = {
    "bert": BertModel,
    "chinese_bert_wwm": BertModel,
    "chatglm": AutoModel,
}


def get_slot_labels(args):
    return [
        label.strip()
        for label in open(
            os.path.join(args.data_dir, args.task, args.slot_label_file),
            "r",
            encoding="utf-8",
        )
    ]


# BertTokenizer.from_pretrained(bert-base-chinese)
def load_tokenizer(args):
    if args.model_type in ["chatglm"]:
        return MODEL_CLASSES[args.model_type][2].from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
    else:
        return MODEL_CLASSES[args.model_type][2].from_pretrained(
            args.model_name_or_path
        )


def read_prediction_text(args):
    return [
        text.strip()
        for text in open(
            os.path.join(args.pred_dir, args.pred_input_file), "r", encoding="utf-8"
        )
    ]


def init_logger(args):
    (root_dir / "logs").mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        filename=f"logs/predict-{args.task}-{args.model_type}-{args.model_dir}.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


# 为当前GPU设置随机种子，以使得结果是确定的
# 不加入manual_seed时，随机数会变化
def set_seed(args):
    torch.manual_seed(args.seed)  # 为cpu分配随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)  # 为gpu分配随机种子
        torch.cuda.manual_seed_all(args.seed)  # 若使用多块gpu，使用该命令设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmard = False


def compute_metrics(slot_preds, slot_labels):
    assert len(slot_preds) == len(slot_labels)
    results = {}
    slot_result = get_slot_metrics(slot_preds, slot_labels)

    results.update(slot_result)
    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds),
    }


def read_prediction_test(args):
    return [
        text.strip()
        for text in open(
            os.path.join(args.pred_dir, args.pred_input_file), "r", encoding="utf-8"
        )
    ]


def read_data(path: str) -> Any:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        data = data.split("\n")
    elif path.endswith(".csv"):
        data = pd.read_csv(path)
    elif path.endswith(".ndjson"):
        data = pd.read_json(path, lines=True, orient="records")
    elif path.endswith(".ndjson.gz"):
        data = pd.read_json(path, lines=True, orient="records", compression="gzip")
    elif path.endswith(".pickle"):
        data = pd.read_pickle(path)
    elif path.endswith(".parquet"):
        data = pd.read_parquet(path)
    elif path.endswith(".yaml"):
        with open(path, "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                logging.error(e)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
    return data


def save_data(data: Any, path: str) -> None:
    if path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    elif path.endswith(".txt") and isinstance(data, list):
        with open(path, "w", encoding="utf-8") as f:
            for _d in data:
                f.write(_d)
                f.write("\n")
    elif path.endswith(".csv"):
        data.to_csv(path, index=False)
    elif path.endswith(".ndjson"):
        data.to_json(path, lines=True, orient="records")
    elif path.endswith(".ndjson.gz"):
        data.to_json(path, lines=True, orient="records", compression="gzip")
    elif path.endswith(".pickle"):
        data.to_pickle(path)
    elif path.endswith(".parquet"):
        data.to_parquet(path)
    elif isinstance(data, list):
        with open(path, "w", encoding="utf-8") as f:
            for _d in data:
                f.write(_d)
                f.write("\n")
    else:
        pass
