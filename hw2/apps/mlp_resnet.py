import sys
sys.path.append("../python")

import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os


np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    # BEGIN YOUR SOLUTION
    return nn.Sequential(nn.Residual(nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim))),
        nn.ReLU(),
    )

    # END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    # BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob)
          for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes),
    )
    # END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    loss_sum = []
    error_sum = []
    sample_num = len(dataloader.dataset)
    if opt is not None:
        model.train()
        for X, y in dataloader:
            logits = model(X)
            loss = nn.SoftmaxLoss()(logits, y)
            loss_sum.append(loss.numpy())
            error_sum.append((logits.numpy().argmax(1) != y.numpy()).sum())
            loss.backward()
            opt.step()
    else:
        model.eval()
        for X, y in dataloader:
            logits = model(X)
            loss = nn.SoftmaxLoss()(logits, y)
            loss_sum.append(loss.numpy())
            error_sum.append((logits.numpy().argmax(1) != y.numpy()).sum())

    return np.sum(error_sum) / sample_num, np.mean(loss_sum)
    # END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    dataloader = ndl.data.DataLoader(
        ndl.data.MNISTDataset(os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
                              os.path.join(data_dir, "train-labels-idx1-ubyte.gz")),
        batch_size=batch_size,
        shuffle=True,
    )
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        train_acc, train_loss = epoch(dataloader, model, opt)
    dataloader = ndl.data.DataLoader(
        ndl.data.MNISTDataset(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
                              os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")),
        batch_size=batch_size,
        shuffle=True,
    )
    test_acc, test_loss = epoch(dataloader, model)
    return train_acc, train_loss, test_acc, test_loss

    # END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
