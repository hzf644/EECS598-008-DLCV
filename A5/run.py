import os
import sys
import time
from transformers import *
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from eecs598.utils import (
    reset_seed,
    tensor_to_image,
    attention_visualizer,
)
from eecs598.grad import rel_error, compute_numeric_gradient
import matplotlib.pyplot as plt
from a5_helper import *
from a5_helper import train as train_transformer
from a5_helper import val as val_transformer
from sklearn.model_selection import train_test_split
from transformers import AddSubDataset
from tqdm import tqdm

plt.rcParams["figure.figsize"] = (10.0, 8.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

GOOGLE_DRIVE_PATH = "D:\\AppData\\AI\\A5"
sys.path.append(GOOGLE_DRIVE_PATH)
DEVICE = torch.device('cuda')
to_float = torch.float
to_long = torch.long

data = get_toy_data(os.path.join(GOOGLE_DRIVE_PATH, "two_digit_op.json"))

"======================================="
SPECIAL_TOKENS = ["POSITIVE", "NEGATIVE", "add", "subtract", "BOS", "EOS"]
vocab = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] + SPECIAL_TOKENS
convert_str_to_tokens = generate_token_dict(vocab)

X, y = data["inp_expression"], data["out_expression"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
inp_seq_len = 9
out_seq_len = 5
num_heads = 4
emb_dim = 32
dim_feedforward = 32
dropout = 0.2
num_enc_layers = 1
num_dec_layers = 1
vocab_len = len(vocab)
BATCH_SIZE = 4
num_epochs=200 #number of epochs
lr=1e-3 #learning rate after warmup
loss_func = CrossEntropyLoss
warmup_interval = None #number of iterations for warmup

model = Transformer(
    num_heads,
    emb_dim,
    dim_feedforward,
    dropout,
    num_enc_layers,
    num_dec_layers,
    vocab_len,
)
train_data = AddSubDataset(
    X_train,
    y_train,
    convert_str_to_tokens,
    SPECIAL_TOKENS,
    emb_dim,
    position_encoding_simple,
)
valid_data = AddSubDataset(
    X_test,
    y_test,
    convert_str_to_tokens,
    SPECIAL_TOKENS,
    emb_dim,
    position_encoding_simple,
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
)
valid_loader = torch.utils.data.DataLoader(
    valid_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
)

small_dataset = torch.utils.data.Subset(
    train_data, torch.linspace(0, len(train_data) - 1, steps=4).long()
)
small_train_loader = torch.utils.data.DataLoader(
    small_dataset, batch_size=4, pin_memory=True, num_workers=1, shuffle=False
)
trained_model = train_transformer(
    model,
    small_train_loader,
    small_train_loader,
    loss_func,
    num_epochs=num_epochs,
    lr=lr,
    batch_size=BATCH_SIZE,
    warmup_interval=warmup_interval,
    device=DEVICE,
)

print(
    "Overfitted accuracy: ",
    "{:.4f}".format(
        val_transformer(
            trained_model,
            small_train_loader,
            CrossEntropyLoss,
            batch_size=4,
            device=DEVICE,
        )[1]
    ),
)
