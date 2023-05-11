import sys
sys.path.append('./')
sys.path.append('..')
from parameters import model, device
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, SubsetRandomSampler
from PIL import Image
from mnist.main import CNN, D1CNN, D4CNN, C2CNN, C4CNN, C8CNN, set_seed
import torch
from tqdm import tqdm
from torchvision.transforms import ToTensor

def load_data():
    X = []
    Y = []
    for i in range(10):
        for d in os.listdir("dataset/assets/{}".format(i)):
            t_img = cv2.imread("dataset/assets/{}".format(i)+"/"+d)
            t_img = cv2.cvtColor(t_img,cv2.COLOR_BGR2GRAY)
            X.append(t_img)
            Y.append(i)

    X = np.array(X, dtype=np.float32) / 255.
    Y = np.array(Y)
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    Y = Y[perm]
    return X, Y

class PrintedDigitDataset(Dataset):
    def __init__(self, X, Y, mode, transform=None):
        assert mode in ['train', 'test', 'valid']
        self.transform = transform
        if mode == 'train':
            self.data = X[:4000].astype(np.float32)
            self.targets = Y[:4000].astype(np.int64)
        elif mode == 'valid':
            self.data = X[4000:5000].astype(np.float32)
            self.targets = Y[4000:5000].astype(np.int64)
        else:
            self.data = X[5000:6000].astype(np.float32)
            self.targets = Y[5000:6000].astype(np.int64)
        self.num_samples = len(self.targets)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def train():
    min_epochs = 50
    max_epochs = 1000
    max_no_improve = 50

    totensor = ToTensor()

    assert model in ['c4', 'c8', 'c2', 'd4', 'cnn', 'd1']

    if model == 'c4':
        network = C4CNN().to(device)
    elif model == 'c8':
        network = C8CNN().to(device)
    elif model == 'c2':
        network = C2CNN().to(device)
    elif model == 'd4':
        network = D4CNN().to(device)
    elif model == 'cnn':
        network = CNN().to(device)
    elif model == 'd1':
        network = D1CNN().to(device)
    else:
        raise NotImplementedError

    X, Y = load_data()

    mnist_train = PrintedDigitDataset(X, Y, mode='train', transform=totensor)
    mnist_valid = PrintedDigitDataset(X, Y, mode='valid', transform=totensor)
    mnist_test = PrintedDigitDataset(X, Y, mode='test', transform=totensor)

    subsample_train_indices = torch.randperm(len(mnist_train.data))
    subsample_valid_indices = torch.randperm(len(mnist_valid.data))
    subsample_test_indices = torch.randperm(len(mnist_test.data))

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=256, sampler=SubsetRandomSampler(subsample_train_indices))
    valid_loader = torch.utils.data.DataLoader(mnist_valid, batch_size=2048, sampler=SubsetRandomSampler(subsample_valid_indices))
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=2048, sampler=SubsetRandomSampler(subsample_test_indices))

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=5e-5, weight_decay=1e-5)

    no_improve = 0
    best_valid_acc = 0
    best_test_acc = 0
    best_test_per_acc = [0 for _ in range(10)]
    best_test_pred_count = np.zeros((10, 10))
    pbar = tqdm(total=max_epochs)
    for epoch in range(max_epochs):
        network.train()
        for i, (x, t) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            t = t.to(device)
            y = network(x)
            loss = loss_function(y, t)
            loss.backward()
            optimizer.step()
        pbar.update()

        valid_total = 0
        valid_correct = 0
        test_total = 0
        test_correct = 0
        test_per_total = [0 for _ in range(10)]
        test_per_correct = [0 for _ in range(10)]
        test_pred_count = np.zeros((10, 10))
        with torch.no_grad():
            network.eval()
            for i, (x, t) in enumerate(valid_loader):
                x = x.to(device)
                t = t.to(device)
                y = network(x)
                _, prediction = torch.max(y.data, 1)
                valid_total += t.shape[0]
                valid_correct += (prediction == t).sum().item()

            for i, (x, t) in enumerate(test_loader):
                x = x.to(device)
                t = t.to(device)

                y = network(x)

                _, prediction = torch.max(y.data, 1)
                test_total += t.shape[0]
                test_correct += (prediction == t).sum().item()

                for j in range(10):
                    test_per_total[j] += (t == j).sum().item()
                    test_per_correct[j] += (prediction == t)[t == j].sum().item()
                    test_pred_count[j] += prediction[t == j].bincount(minlength=10).cpu().numpy()

        test_per_acc = (np.array(test_per_correct) / np.array(test_per_total) * 100.).round(2).tolist()
        test_per_acc_dic = {}
        for j in range(10):
            test_per_acc_dic[j] = test_per_acc[j]

        test_acc = np.round(test_correct / test_total * 100., 2)
        valid_acc = test_correct / test_total
        if valid_acc > best_valid_acc:
            no_improve = 0
            best_valid_acc = valid_acc
            best_test_acc = test_acc
            best_test_per_acc = test_per_acc_dic
            best_test_pred_count = test_pred_count
        else:
            no_improve += 1
        pbar.set_description('epoch: {}, valid: {}, no improve: {}, best test: {}'.
                             format(epoch, valid_acc, no_improve, best_test_acc))
        pbar.update()
        if no_improve >= max_no_improve and epoch > min_epochs:
            break
    pbar.close()
    return best_test_acc, best_test_per_acc, best_test_pred_count

if __name__ == '__main__':
    # np.set_printoptions(threshold=np.inf, linewidth=200)
    all_best = []
    all_best_per = []
    all_pred_count = np.zeros((10, 10))
    for seed in range(10):
    # for seed in [1]:
        set_seed(seed)
        best, best_per, pred_count = train()
        all_best.append(best)
        all_best_per.append(list(best_per.values()))
        all_pred_count += pred_count
        all_best_per_avg = np.array(all_best_per).mean(0).round(2).tolist()
        print(f'{np.mean(all_best).round(2)}, {all_best_per_avg}')
        # print(all_pred_count.astype(int))
        for row in all_pred_count.astype(int):
            print(row.tolist())