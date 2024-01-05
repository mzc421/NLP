import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from seqeval.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

from datasets import IntentSlotDataset
from models import JointBert
from tools import save_module, split_data


def dev(model, val_dataloader, device, slot_dict):
    model.eval()
    intent_acc, slot_acc = 0, 0
    all_true_intent, all_pred_intent = [], []
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            input_ids, intent_labels, slot_labels = batch

            outputs = model(
                input_ids=torch.tensor(input_ids).long().to(device),
                intent_labels=torch.tensor(intent_labels).long().to(device),
                slot_labels=torch.tensor(slot_labels).long().to(device)
            )

            intent_probs = torch.softmax(outputs["intent_logits"], dim=-1).detach().cpu().numpy()
            slot_probs = torch.softmax(outputs["slot_logits"], dim=-1).detach().cpu().numpy()
            slot_ids = np.argmax(slot_probs, axis=-1)
            intent_ids = np.argmax(intent_probs, axis=-1)
            slot_ids = slot_ids.tolist()
            intent_ids = intent_ids.tolist()

            slot_ids = [[slot_dict[i] for i in line] for line in slot_ids]
            slot_labels = [[slot_dict[i] for i in line] for line in slot_labels]

            all_true_intent.extend(intent_labels)
            all_pred_intent.extend(intent_ids)

            intent_acc += accuracy_score(intent_labels, intent_ids)
            slot_acc += accuracy_score(slot_labels, slot_ids)

    intent_avg, slot_avg = intent_acc / len(val_dataloader), slot_acc / len(val_dataloader)
    dev_acc = intent_avg + slot_avg
    return dev_acc, intent_avg, slot_avg


def train(args):
    # 模型保存位置
    model_save_dir = args.save_dir + "/" + args.model_path.split("/")[-1]

    with open(args.slot_label_path, 'r') as f:
        slot_labels = f.read().strip('\n').split('\n')
    slot_dict = dict(zip(range(len(slot_labels)), slot_labels))

    # -----------set cuda environment-------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------load tokenizer-----------
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    save_module(tokenizer, model_save_dir)

    # -----------load data-----------------
    train_data, val_data = split_data(args.train_data_path, args.train_val_data_split)

    train_dataset = IntentSlotDataset.load_from_path(
        data_content=train_data,
        intent_label_path=args.intent_label_path,
        slot_label_path=args.slot_label_path,
        tokenizer=tokenizer
    )

    val_dataset = IntentSlotDataset.load_from_path(
        data_content=val_data,
        intent_label_path=args.intent_label_path,
        slot_label_path=args.slot_label_path,
        tokenizer=tokenizer
    )

    # -----------load model and dataset-----------
    model = JointBert.from_pretrained(
        args.model_path,
        slot_label_num=train_dataset.slot_label_num,
        intent_label_num=train_dataset.intent_label_num
    )
    model = model.to(device).train()

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=train_dataset.batch_collate_fn)

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=val_dataset.batch_collate_fn)

    # -----------calculate training steps-----------
    if args.max_training_steps > 0:
        total_steps = args.max_training_steps
    else:
        total_steps = len(train_dataset) * args.train_epochs // args.gradient_accumulation_steps // args.batch_size

    print('calculated total optimizer update steps : {}'.format(total_steps))

    # -----------prepare optimizer and schedule------------
    parameter_names_no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # 这些参数会被应用常规的权重衰减（由 args.weight_decay 指定）
        {'params': [
            para for para_name, para in model.named_parameters()
            if not any(nd_name in para_name for nd_name in parameter_names_no_decay)
        ],
            'weight_decay': args.weight_decay},
        # 这些参数的权重衰减被设置为0
        {'params': [
            para for para_name, para in model.named_parameters()
            if any(nd_name in para_name for nd_name in parameter_names_no_decay)
        ],
            'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # 学习率变化（更新）方式
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    # -----------training-------------
    max_acc = 0
    for epoch in range(args.train_epochs):
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            input_ids, intent_labels, slot_labels = batch

            outputs = model(
                input_ids=torch.tensor(input_ids).long().to(device),
                intent_labels=torch.tensor(intent_labels).long().to(device),
                slot_labels=torch.tensor(slot_labels).long().to(device)
            )

            loss = outputs['loss']
            total_loss += loss.item()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if step % args.gradient_accumulation_steps == 0:
                # 用于对梯度进行裁剪，以防止在神经网络训练过程中出现梯度爆炸的问题。
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()

        train_loss = total_loss / len(train_dataloader)

        dev_acc, intent_avg, slot_avg = dev(model, val_dataloader, device, slot_dict)

        flag = False
        if max_acc < dev_acc:
            max_acc = dev_acc
            flag = True
            save_module(model, model_save_dir)
        print(f"[{epoch}/{args.train_epochs}] train loss: {train_loss}  dev intent_avg: {intent_avg} "
              f"def slot_avg: {slot_avg} save best model: {'*' if flag else ''}")

    dev_acc, intent_avg, slot_avg = dev(model, val_dataloader, device, slot_dict)
    print("last model dev intent_avg: {} def slot_avg: {}".format(intent_avg, slot_avg))
    print("模型保存位置：" + model_save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # environment parameters
    parser.add_argument("--cuda_devices", type=str, default='0', help='set cuda device numbers')

    # model parameters
    parser.add_argument("--model_path", type=str, default='./bert-base-chinese', help="pretrained model loading path")

    # data parameters
    parser.add_argument("--train_data_path", type=str, default='data/SMP2019/data.json', help="training data path")
    parser.add_argument("--train_val_data_split", type=float, default=0.8, help="training data and val data split rate")
    parser.add_argument("--slot_label_path", type=str, default='data/SMP2019/slot_labels.txt', help="slot label path")
    parser.add_argument("--intent_label_path", type=str, default='data/SMP2019/intent_labels.txt', help="intent label path")

    # training parameters
    parser.add_argument("--save_dir", type=str, default='./save_model', help="directory to save the model")
    parser.add_argument("--max_training_steps", type=int, default=0, help='max training step for optimizer(优化器的最大训练步数), if larger than 0')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="number of updates steps to accumulate before performing a backward() pass.(执行 backward() 之前累积的更新步数数量)")

    parser.add_argument("--batch_size", type=int, default=32, help='training data batch size')
    parser.add_argument("--train_epochs", type=int, default=20, help='training epoch number')

    parser.add_argument("--learning_rate", type=float, default=5e-5, help='learning rate')
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="epsilon for Adam optimizer")
    parser.add_argument("--warmup_steps", type=int, default=0, help="warmup step number")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="maximum norm for gradients")

    args = parser.parse_args()

    train(args)
