import sys
sys.path.append('./')
sys.path.append('..')
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from tqdm import tqdm

from square_networks import MLPDSSX, MLPNoX
from square_parameters import *
from logger import Logger

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getData(mr=0., cr=0., plot=False):
    assert 0 <= mr <= 1
    assert 0 <= cr <= 1
    x = np.linspace(0, 1, 1000)
    y = np.linspace(0, 1, 1000)
    X, Y = np.meshgrid(x, y)
    data = np.array([X.flatten(), Y.flatten()]).T
    label = np.zeros(data.shape[0])
    label[data[:, 1] < cr] = 0
    for i in range(5):
        if i * 0.2 < mr:
            l = 1
        else:
            l = i+1
        label[(data[:, 1] >= cr) * (data[:, 0] >= i * 0.2) * (data[:, 0] <= (i+1) * 0.2)] = l
    if plot:
        fig = plt.figure(dpi=300, figsize=(5, 5))
        plt.plot(data[label==0, 0], data[label==0, 1], 'o', color='r', markersize=0.1)
        plt.plot(data[label==1, 0], data[label==1, 1], 'o', color='g', markersize=0.1)
        plt.plot(data[label==2, 0], data[label==2, 1], 'o', color='b', markersize=0.1)
        plt.plot(data[label==3, 0], data[label==3, 1], 'o', color='purple', markersize=0.1)
        plt.plot(data[label==4, 0], data[label==4, 1], 'o', color='orange', markersize=0.1)
        plt.plot(data[label==5, 0], data[label==5, 1], 'o', color='black', markersize=0.1)
        plt.tight_layout()
        plt.show()

    data = torch.tensor(data, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long)
    permutation = torch.tensor(np.random.permutation(data.shape[0]))
    data = data[permutation]
    label = label[permutation]

    return data, label

def train():
    data, label = getData(mr=mr, cr=cr)
    n_train = n_data
    n_holdout = 200
    min_epochs = 1000
    max_epochs_no_improve = 1000
    max_epochs = 10000
    batch_size = 128

    # if model == 'mlp':
    #     network = MLP(3, 2).to(device)
    if model == 'invx':
        network = MLPNoX(6).to(device)
    elif model == 'dssx':
        network = MLPDSSX(6).to(device)
    else:
        raise NotImplementedError
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    log_dir = os.path.join(log_pre, 'square_{}'.format(model)) + '_ndata{}_mr{}_cr{}'.format(n_data, mr, cr)
    logger = Logger(log_dir, model, seed=args.seed)

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
    del data, label

if __name__ == '__main__':
    global mr, cr, model

    # getData(0.2, 0.2, plot=True)
    for m, c in [(0.2, 0), (0.2, 0.2), (0.2, 0.4), (0.2, 0.6), (0.2, 0.8),
                 (0.4, 0), (0.4, 0.2), (0.4, 0.4), (0.4, 0.6), (0.4, 0.8),
                 (0.6, 0), (0.6, 0.2), (0.6, 0.4), (0.6, 0.6), (0.6, 0.8),
                 (0.8, 0), (0.8, 0.2), (0.8, 0.4), (0.8, 0.6), (0.8, 0.8),
                 (1, 0), (1, 0.2), (1, 0.4), (1, 0.6), (1, 0.8), (1, 1)
                 ]:
        mr, cr = m, c
        for s in range(10):
            args.seed = s
            set_seed(s)
            train()
            torch.cuda.empty_cache()