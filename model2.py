import os

import click
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TokenClassificationPipeline,
    Trainer,
    TrainingArguments,
)
from utils import convertor_zhtw_zhcn, read_data, root_dir, save_data


@click.command()
@click.option(
    "--task",
    "task",
    type=str,
    default="gpt35_filtered",
)
@click.option(
    "--pretrained",
    "pretrained",
    type=str,
    default="bert-base-chinese",
)
def main(task: str, pretrained: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    slot_labels = read_data(
        path=str(root_dir / "data" / "BIOES" / task / "slot_label.txt")
    )
    label2id = {k: i for i, k in enumerate(slot_labels)}
    id2label = {i: k for i, k in enumerate(slot_labels)}
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    seqeval = evaluate.load("seqeval")

    train_in = read_data(
        path=str(root_dir / "data" / "BIOES" / task / "train" / "seq.in")
    )
    train_out = read_data(
        path=str(root_dir / "data" / "BIOES" / task / "train" / "seq.out")
    )
    dev_in = read_data(path=str(root_dir / "data" / "BIOES" / task / "dev" / "seq.in"))
    dev_out = read_data(
        path=str(root_dir / "data" / "BIOES" / task / "dev" / "seq.out")
    )
    test_in = read_data(
        path=str(root_dir / "data" / "BIOES" / task / "test" / "seq.in")
    )
    test_out = read_data(
        path=str(root_dir / "data" / "BIOES" / task / "test" / "seq.out")
    )
    output_dir = f"{task}-{pretrained}"

    ds = Dataset.from_dict(
        {
            "id": [
                str(idx)
                for idx in range(
                    len(train_in.split("\n"))
                    + len(dev_in.split("\n"))
                    + len(test_in.split("\n"))
                )
            ],
            "tokens": (train_in + "\n" + dev_in + "\n" + test_in).split("\n"),
            "ner_tags": (train_out + "\n" + dev_out + "\n" + test_out).split("\n"),
        }
    )
    ds_train = ds.train_test_split(test_size=0.3)
    ds_dev_test = ds_train["test"].train_test_split(test_size=0.3)
    ds = DatasetDict(
        {
            "train": ds_train["train"],
            "dev": ds_dev_test["train"],
            "test": ds_dev_test["test"],
        }
    )

    def tokenize_and_align_labels(examples):
        # tag name to id
        examples["tokens"] = [piece.split() for piece in examples["tokens"]]
        examples["ner_tags"] = [
            [label2id[tag] for tag in piece.split()] for piece in examples["ner_tags"]
        ]

        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [slot_labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [slot_labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def compute_metrics_by_labels(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [slot_labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [slot_labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return results

    tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
    model = AutoModelForTokenClassification.from_pretrained(
        pretrained,
        num_labels=len(slot_labels),
        id2label=id2label,
        label2id=label2id,
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        resume_from_checkpoint=1110,
        # overwrite_output_dir=True,
        # learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        # num_train_epochs=20,
        # weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # save_total_limit=1,
        seed=5408,
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_by_labels,
    )
    # trainer.train()
    testset_metrics = trainer.evaluate(tokenized_ds["test"])
    print(testset_metrics)
    save_data(testset_metrics, f"{task}-{pretrained}/testset_metrics.json")

    # get prediction labels through pipeline
    # _testset_text_list = ["".join(l) for l in tokenized_ds["test"]["tokens"]]
    # finetuned_path = [p for p in os.listdir(output_dir) if "checkpoint" in p][0]
    # finetuned = AutoModelForTokenClassification.from_pretrained(
    #     root_dir / output_dir / finetuned_path,
    #     num_labels=len(slot_labels),
    #     id2label=id2label,
    #     label2id=label2id,
    # )
    # pipe = TokenClassificationPipeline(model=finetuned, tokenizer=tokenizer)
    # predict_output = pipe(_testset_text_list)
    # output_tags = []
    # for i, _set in tqdm(enumerate(predict_output)):
    #     _input_text = _testset_text_list[i]
    #     _output_tags = ["O"] * len(_input_text)
    #     _pred_lb_pos = {}
    #     for _lb in _set:
    #         _pred_lb_pos[_lb["start"]] = _lb["entity"]
    #     for lb_idx in _pred_lb_pos:
    #         _output_tags[lb_idx] = _pred_lb_pos[lb_idx]
    #     output_tags.append(_output_tags)

    # with open(f"{output_dir}/testset_outputs.txt", "w", encoding="utf-8") as f:
    #     for i in range(len(output_tags)):
    #         f.write(f'Text: {" ".join(tokenized_ds["test"]["tokens"][i])}\n')
    #         f.write(f'Slot: {" ".join(output_tags[i])}\n')
    #         f.write("\n")

    # predict_output = trainer.predict(tokenized_ds["test"])
    # print(predict_output)
    # save_data(
    #     pd.DataFrame([predict_output._asdict()]),
    #     f"{task}-{pretrained}/testset_output.pickle",
    # )


if __name__ == "__main__":
    main()

# from datasets import load_dataset

# wnut = load_dataset("wnut_17")
# wnut["train"][0]
# wnut["train"].features["ner_tags"].feature.names
# tokenized_input = tokenizer(wnut["train"][0]["tokens"], truncation=True, is_split_into_words=True)
# tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
