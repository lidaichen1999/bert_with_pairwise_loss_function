#!/usr/bin/env python
# coding: utf-8

# In[1]:

# ### 1. 安装所需要的包

# In[2]:


# !pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
# !pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
# !pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
# !pip install scikit_learn -i https://pypi.tuna.tsinghua.edu.cn/simple


# In[3]:


import json
import logging
import random
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import requests
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from transformers import AdamW, BertModel, BertPreTrainedModel, BertTokenizer
import nltk
logging.basicConfig(
    level=logging.INFO,
    filename=f"bert_pair.log",
    filemode="w",
    format="%(asctime)s - %(message)s",
)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# In[4]:


SEED = 9999
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# ### 2.分词器，wordpiece

# In[5]:


# huggingface
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# ### 3.读入数据

# In[6]:


df = pd.read_csv("all_added.tsv_changes.tsv", sep="\t")


# In[7]:


df


# In[8]:


# 找同义或者近义词
def get_synonims(word):
    base_url = 'https://api.datamuse.com/words'

    def list_of_words(word, base_url=base_url):
        query_dict = dict()
        query_dict['rel_syn'] = word
        query_dict['max'] = 5
        response = requests.get(base_url, params=query_dict)
        data = response.json()
        words = [d['word'] for d in data]
        if len(words) < 1:
            return None
        return words

    syns = list_of_words(word)
    if syns is None:
        return []
    else:
        return syns


# In[9]:


all_labels = list(set(df["before_pt"]) | set(df["pt"]))


# In[10]:


def augmentation(text):
    new_texts = None
    split_text = nltk.word_tokenize(text)
    for jdx, (word, flag) in enumerate(nltk.pos_tag(split_text)):
        if not flag.startswith('N'):
            continue
        res = get_synonims(word)
        if len(res)!=0:
            split_text[jdx] = res[0]
            new_texts = " ".join(split_text)
            break
    return new_texts


# In[12]:


def read_data(file, aug=2, do_augment=False):
    texts = []
    labels = []
    data = pd.read_csv(file, sep="\t")
    for idx in tqdm(data.itertuples(), total=len(data)):
        text = str(getattr(idx, "before_anno"))
        text = re.sub("\[.*?\]", "", text)
        text = re.sub(" {2,}", " ", text).strip()
        label = getattr(idx, "before_pt")
        neg_label = getattr(idx, "pt")
        if pd.isna(label) or pd.isna(neg_label):
            continue
        texts.append(text)
        labels.append([label, neg_label])
        if do_augment:
            new_texts = augmentation(text)
            if new_texts is not None:
                texts.append(new_texts)
                labels.append(labels[-1])
        temp = [label, neg_label]
        for i in range(aug):
            ran = random.choice(all_labels)
            while ran in temp:
                ran = random.choice(all_labels)
            texts.append(text)
            labels.append([label, ran])
            temp.append(ran)
            if do_augment:
                new_texts = augmentation(text)
                if new_texts is not None:
                    texts.append(new_texts)
                    labels.append(labels[-1])

    assert len(texts) == len(labels)
    return texts, labels


# In[13]:


texts, labels = read_data("all_added.tsv_changes.tsv", 2, False)


# In[14]:


df = pd.DataFrame()
df['texts'] = texts
df['labels'] = labels
df.to_csv('result.csv',index=False, encoding='utf-8-sig')


# In[15]:


len(texts)


# ### 4.训练集和验证集

# In[16]:


train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.3, random_state=43
)


# In[17]:


len(train_labels), len(val_labels), len(train_texts), len(val_texts)


# ### 5.查看text和label

# In[18]:


train_texts[:3]


# In[19]:


train_labels[:3]


# ### 6.求最大长度，为后面分词做准备

# In[20]:


max_len = max([len(item) for item in train_texts])
print(max_len)

max_len = max([len(item) for item in val_texts])
print(max_len)


# ### 7. label和id进行映射

# In[21]:


label2id = {item: idx for idx, item in enumerate(sorted(set(all_labels)))}
id2label = {v: k for k, v in label2id.items()}


# In[22]:


label2id, id2label


# ### 8.训练集和验证集 分词

# In[23]:


train_pos = []
train_neg = []
for i, j in zip(train_texts, train_labels):
    train_pos.append([i, j[0]])
    train_neg.append([i, j[1]])


# In[24]:


val_pos = []
val_neg = []
for i, j in zip(val_texts, val_labels):
    val_pos.append([i, j[0]])
    val_neg.append([i, j[1]])


# In[25]:


train_pos_encodings = tokenizer(
    [i for i, j in train_pos],
    [j for i, j in train_pos],
    truncation=True,
    padding="max_length",
    max_length=128,
    truncation_strategy="only_first", 
)


# In[26]:


train_neg_encodings = tokenizer(
    [i for i, j in train_neg],
    [j for i, j in train_neg],
    truncation=True,
    padding="max_length",
    max_length=128,
    truncation_strategy="only_first", 
)


# In[27]:


val_pos_encodings = tokenizer(
    [i for i, j in val_pos],
    [j for i, j in val_pos],
    truncation=True,
    padding="max_length",
    max_length=128,
    truncation_strategy="only_first", 
)


# In[28]:


val_neg_encodings = tokenizer(
    [i for i, j in val_neg],
    [j for i, j in val_neg],
    truncation=True,
    padding="max_length",
    max_length=128,
    truncation_strategy="only_first", 
)


# ### 9.创建Dataset

# In[29]:


# PyTorch Dataset
class CuDataset(torch.utils.data.Dataset):
    def __init__(self, pos_encodings, neg_encodings):
        self.pos_encodings = pos_encodings
        self.neg_encodings = neg_encodings

    def __getitem__(self, idx):
        idx = int(idx)
        item = {
            key + "_pos": torch.tensor(val[idx])
            for key, val in self.pos_encodings.items()
        }
        item.update(
            {
                key + "_neg": torch.tensor(val[idx])
                for key, val in self.neg_encodings.items()
            }
        )
        return item

    def __len__(self):
        return len(self.pos_encodings["input_ids"])


# In[30]:


train_dataset = CuDataset(train_pos_encodings, train_neg_encodings)
val_dataset = CuDataset(val_pos_encodings, val_neg_encodings)


# ### 10.创建Dataloader

# In[31]:


batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# ### 11.加载模型

# In[32]:


class PairModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_states = outputs[1]
        values = self.classifier(last_hidden_states)
        value = values.mean(dim=1).squeeze(-1)
        return values


# In[33]:


model = PairModel.from_pretrained("bert-base-uncased")

device = (
    torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
)  # 使用cpu或者gpu
model.to(device)
model.train()


# In[34]:


len(train_loader)


# ### 12.计算Accuracy，Precision，Recall，F1 score

# In[35]:


def compute_metrics(all_cnt, cor_cnt):
    accuracy = cor_cnt / all_cnt
    logging.info(f"accuracy: {accuracy}")
    return accuracy


# In[36]:


def compute_metrics_ano(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds,average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    logging.info(f'accuracy: {accuracy}')
    logging.info(f'precision: {precision}')
    logging.info(f'recall: {recall}')
    logging.info(f'f1: {f1}\n')
    return accuracy, precision, recall, f1


# ### 13.评估模型

# In[37]:


@torch.no_grad()
def eval_model_ano(model, eval_loader):
    model.eval()
    all_cnt = []
    cor_cnt = []
    for idx, batch in enumerate(eval_loader):
        pos_input_ids = batch["input_ids_pos"].to(device)
        pos_attention_mask = batch["attention_mask_pos"].to(device)
        pos_token_type_ids = batch["token_type_ids_pos"].to(device)
        pos_outputs = model(
            input_ids=pos_input_ids,
            attention_mask=pos_attention_mask,
            token_type_ids=pos_token_type_ids,
        )

        neg_input_ids = batch["input_ids_neg"].to(device)
        neg_attention_mask = batch["attention_mask_neg"].to(device)
        neg_token_type_ids = batch["token_type_ids_neg"].to(device)
        neg_outputs = model(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            token_type_ids=neg_token_type_ids,
        )
        for i in range(len(pos_outputs)):
            all_cnt.append(1)
            if pos_outputs[i] > neg_outputs[i]:
                cor_cnt.append(1)
            else:
                cor_cnt.append(0)
    accuracy, precision, recall, f1 = compute_metrics_ano(all_cnt, cor_cnt)
    model.train()
    return accuracy, precision, recall, f1


# ### 14.训练模型

# In[38]:


class LogSigLoss(nn.Module):
    """
    Pairwise Loss
    Details: https://arxiv.org/abs/2203.02155
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.sigmoid(chosen_reward - reject_reward)
        log_probs = torch.log(probs)
        loss = -log_probs.mean()
        return loss


# In[39]:


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss


# In[40]:


loss_fn = LogSigLoss()


# In[ ]:


param_optimizer = list(model.named_parameters())
no_decay = ["bias", "gamma", "beta"]
optimizer_grouped_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay_rate": 0.01,
    },
    {
        "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        "weight_decay_rate": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

step = 0
best_acc = 0
epoch = 10
model_path = f"model_best"
writer = SummaryWriter(log_dir=model_path)
for epoch in tqdm(range(epoch)):
    for idx, batch in tqdm(
        enumerate(train_loader),
        total=len(train_pos) // batch_size,
        leave=False,
    ):
        optimizer.zero_grad()
        pos_input_ids = batch["input_ids_pos"].to(device)
        pos_attention_mask = batch["attention_mask_pos"].to(device)
        pos_token_type_ids = batch["token_type_ids_pos"].to(device)
        pos_outputs = model(
            input_ids=pos_input_ids,
            attention_mask=pos_attention_mask,
            token_type_ids=pos_token_type_ids,
        )

        neg_input_ids = batch["input_ids_neg"].to(device)
        neg_attention_mask = batch["attention_mask_neg"].to(device)
        neg_token_type_ids = batch["token_type_ids_neg"].to(device)
        neg_outputs = model(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            token_type_ids=neg_token_type_ids,
        )
        loss = loss_fn(pos_outputs, neg_outputs)
        logging.info(f"Epoch-{epoch}, Step-{step}, Loss: {loss.cpu().detach().numpy()}")
        step += 1
        loss.backward()
        optimizer.step()
        writer.add_scalar("train_loss", loss.item(), step)
    logging.info(f"Epoch {epoch}, present best acc: {best_acc}, start evaluating.")
    accuracy, precision, recall, f1 = eval_model_ano(model, eval_loader)  # 评估模型
    writer.add_scalar("dev_accuracy", accuracy, step)
    if accuracy > best_acc:
        model.save_pretrained(model_path)  # 保存模型
        tokenizer.save_pretrained(model_path)
        best_acc = accuracy

