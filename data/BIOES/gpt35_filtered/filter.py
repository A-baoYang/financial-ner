from collections import Counter
from pathlib import Path

import pandas as pd
from utils import read_data, save_data


def add2dicts(dict1: dict, dict2: dict) -> dict:
    dict3 = {**dict1, **dict2}
    for k, v in dict1.items():
        if k in dict2:
            dict3[k] += v
    return dict3


def count_label_stats(formatted_label_data: pd.DataFrame) -> dict:
    label_stats = {}
    for _l in formatted_label_data["openai_label_bioes"]:
        label_stats = add2dicts(label_stats, dict(Counter(_l)))

    print(label_stats)
    return label_stats


formatted_label_data = read_data("formatting_result.ndjson.gz")

# split train/val/test
# 排除沒有任何實體的樣本
filtered_formatted_label_data = formatted_label_data[
    formatted_label_data["openai_label_offset"].apply(lambda x: True if x else False)
].reset_index(drop=True)
print(len(filtered_formatted_label_data))
label_stats = count_label_stats(filtered_formatted_label_data)
save_data(label_stats, path="filtered_formatting_result_stats.json")

_train, _dev, _test = (
    filtered_formatted_label_data.iloc[: len(filtered_formatted_label_data) // 10 * 7],
    filtered_formatted_label_data.iloc[
        len(filtered_formatted_label_data)
        // 10
        * 7 : len(filtered_formatted_label_data)
        // 10
        * 9
    ].reset_index(drop=True),
    filtered_formatted_label_data.iloc[
        len(filtered_formatted_label_data) // 10 * 9 :
    ].reset_index(drop=True),
)
print(_train, _dev, _test)
for _d, folder_name in zip([_train, _dev, _test], ["train", "dev", "test"]):
    Path(folder_name).mkdir(exist_ok=True, parents=True)
    for i in range(len(_d)):
        assert len(_d["input"][i]) == len(_d["openai_label_bioes"][i])

    _input = [" ".join(_d["input"][i]) for i in range(len(_d))]
    _output = [" ".join(_d["openai_label_bioes"][i]) for i in range(len(_d))]

    save_data(data=_input, path=f"{folder_name}/seq.in")
    save_data(data=_output, path=f"{folder_name}/seq.out")
