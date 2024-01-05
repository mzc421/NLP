import json
import os
import random

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_module(model, save_dir):
    check_path(save_dir)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(save_dir)
    

def split_data(data_path, split_rate):
    with open(data_path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    train_data, val_data = [], []
    intent = {}
    for item in data:
        if item['intent'] not in intent.keys():
            intent[item['intent']] = [item]
        else:
            intent[item['intent']].append(item)

    for key, value in intent.items():
        train = value[:int(len(value) * split_rate)]
        val = value[int(len(value) * split_rate):]
        train_data.extend(train)
        val_data.extend(val)

    random.shuffle(train_data)
    random.shuffle(val_data)
    return train_data, val_data
