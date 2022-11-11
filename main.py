import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch
import os
import torch.nn.functional as F
from tqdm import tqdm

from networks import MLP, MLPNoZ
from parameters import *
from logger import Logger

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getData(icr=0., cr=0.):
    # r = np.arange(0, .09, 0.00001)
    # nverts = len(r)
    # theta = np.array(range(nverts)) * (2*np.pi)/(nverts-1)
    # theta = 90*np.pi*r
    # yy_1 = 10*r * np.sin(theta)
    # xx_1 = 10*r * np.cos(theta)
    # data_1 = np.stack((xx_1, yy_1, np.ones_like(xx_1) * 0), 1)
    #
    # yy_2 = 10*r * np.sin(theta + 3)
    # xx_2 = 10*r * np.cos(theta + 3)
    # data_2 = np.stack((xx_2, yy_2, np.ones_like(xx_1) * 1), 1)
    #
    # data_1 = torch.tensor(data_1)
    # label_1 = torch.ones(data_1.shape[0]) * 0
    # data_2 = torch.tensor(data_2)
    # label_2 = torch.ones(data_2.shape[0]) * 1
    #
    # data = torch.cat([data_1, data_2], dim=0).float()
    # label = torch.cat([label_1, label_2], dim=0).long()
    #
    # permutation = torch.tensor(np.random.permutation(data.shape[0]))
    # data = data[permutation]
    # label = label[permutation]
    extr = 1 - icr - cr
    assert 0 <= extr <= 1
    assert 0 <= icr <= 1
    assert 0 <= cr <= 1
    r = np.arange(0, .09, 0.00001)
    nverts = len(r)
    theta = np.array(range(nverts)) * (2 * np.pi) / (nverts - 1)
    theta = 90 * np.pi * r
    yy_1 = 10 * r * np.sin(theta)
    xx_1 = 10 * r * np.cos(theta)
    data_1 = np.stack((xx_1, yy_1, np.ones_like(xx_1) * 0), 1)

    yy_2 = 10 * r * np.sin(theta + 3)
    xx_2 = 10 * r * np.cos(theta + 3)
    data_2 = np.stack((xx_2, yy_2, np.ones_like(xx_1) * 1), 1)

    permutation = np.arange(len(r))
    n_c = len(r) * cr
    n_ic = len(r) * icr
    n_expt = len(r) - n_c - n_ic

    id_c_1 = permutation[:int(n_c//2)]
    id_c_2 = permutation[int(n_c//2):int(n_c)]
    id_ic = permutation[int(n_c):int(n_c + n_ic)]
    # id_expt = permutation[n_c + n_ic:]

    data_1[id_c_1, :2] = data_1[id_c_2, :2]
    data_1[id_c_1, 2] = 0
    data_1[id_c_2, 2] = 1

    data_2[id_c_2, :2] = data_2[id_c_1, :2]
    data_2[id_c_1, 2] = 0
    data_2[id_c_2, 2] = 1

    # for incorrect data, set the xy to be the same
    data_2[id_ic, :2] = data_1[id_ic, :2]

    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # axs[0].plot(data_1[data_1[:, 2] == 0][:, 0], data_1[data_1[:, 2] == 0][:, 1], 'o', color='r', markersize=0.1)
    # axs[0].plot(data_2[data_2[:, 2] == 0][:, 0], data_2[data_2[:, 2] == 0][:, 1], 'o', color='g', markersize=0.1)
    # axs[1].plot(data_1[data_1[:, 2] == 1][:, 0], data_1[data_1[:, 2] == 1][:, 1], 'o', color='r', markersize=0.1)
    # axs[1].plot(data_2[data_2[:, 2] == 1][:, 0], data_2[data_2[:, 2] == 1][:, 1], 'o', color='g', markersize=0.1)
    #
    # axs[2].plot(data_1[:, 0], data_1[:, 1], 'o', color='r', markersize=0.1)
    # axs[2].plot(data_2[:, 0], data_2[:, 1], 'o', color='g', markersize=0.1)
    # plt.tight_layout()
    # plt.show()

    data_1 = torch.tensor(data_1)
    label_1 = torch.ones(data_1.shape[0]) * 0
    data_2 = torch.tensor(data_2)
    label_2 = torch.ones(data_2.shape[0]) * 1

    data = torch.cat([data_1, data_2], dim=0).float()
    label = torch.cat([label_1, label_2], dim=0).long()

    permutation = torch.tensor(np.random.permutation(data.shape[0]))
    data = data[permutation]
    label = label[permutation]
    return data, label

def train():
    data, label = getData(icr=icr, cr=cr)
    n_train = n_data
    n_holdout = 200
    min_epochs = 1000
    max_epochs_no_improve = 1000
    max_epochs = 10000
    batch_size = 128

    if model == 'mlp':
        network = MLP(3, 2).to(device)
    elif model == 'invz':
        network = MLPNoZ(2).to(device)
    else:
        raise NotImplementedError
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    log_dir = os.path.join(log_pre, '{}'.format(model)) + '_ndata{}_cr{}_icr{}'.format(n_data, cr, icr)
    logger = Logger(log_dir, model)

    train_data = data[:n_train]
    train_label = label[:n_train]
    valid_data = data[n_train:n_train + n_holdout]
    valid_label = label[n_train:n_train + n_holdout]
    test_data = data[n_train + n_holdout:n_train + 2 * n_holdout]
    test_label = label[n_train + n_holdout:n_train + 2 * n_holdout]

    min_valid_loss = 1e10
    epochs_no_improve = 0
    min_test_err = 1e10

    pbar = tqdm(total=max_epochs)
    for epoch in range(1, max_epochs + 1):
        train_idx = np.random.permutation(train_data.shape[0])
        # it = tqdm(range(0, train_data.shape[0], batch_size))
        it = range(0, train_data.shape[0], batch_size)
        for start_pos in it:
            idx = train_idx[start_pos: start_pos + batch_size]
            batch = train_data[idx]
            label_batch = train_label[idx]
            batch = batch.to(device)
            label = label_batch.to(device)
            out = network(batch)
            loss = F.cross_entropy(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.model_losses.append(loss.item())

        with torch.no_grad():
            network.eval()
            valid_data = valid_data.to(device)
            valid_out = network(valid_data)
            # valid_loss = F.cross_entropy(valid_out, valid_gc.to(device))
            valid_loss = 1 - (valid_out.argmax(1) == valid_label.to(device)).sum() / valid_out.shape[0]
            valid_loss = valid_loss.item()

            test_data = test_data.to(device)
            test_out = network(test_data)
            acc = (test_out.argmax(1) == test_label.to(device)).sum() / test_out.shape[0]
            test_err = acc.item()
            network.train()
        logger.model_holdout_losses.append((valid_loss, test_err))
        logger.saveModelLosses()

        if valid_loss < min_valid_loss:
            epochs_no_improve = 0
            min_valid_loss = valid_loss
            min_test_err = test_err
        else:
            epochs_no_improve += 1
        pbar.set_description('epoch: {}, valid loss: {:.03f}, no improve: {}, test sr: {:.03f} ({:.03f})'
                                 .format(epoch, valid_loss, epochs_no_improve, test_err, min_test_err))
        pbar.update()
        if epochs_no_improve >= max_epochs_no_improve and epoch > min_epochs:
            break
    pbar.close()
    logger.saveModelLossCurve()
    logger.saveModelHoldoutLossCurve()
    del network

if __name__ == '__main__':
    # getData(0, 0)
    # getData(0, 0.5)
    # getData(0, 1)
    # getData(0.5, 0)
    # getData(1, 0)
    # getData(0.5, 0.5)

    global cr, icr, model
    for m in ['mlp', 'invz']:
        model = m
        # for c, i in [(0, 0), (0, 0.25), (0, 0.5), (0, 0.75), (0, 1), (0.25, 0), (0.5, 0), (0.75, 0), (1, 0)]:
        for c, i in [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]:
            cr, icr = c, i

            for s in range(0, 4):
                args.seed = s
                set_seed(s)
                train()
                torch.cuda.empty_cache()