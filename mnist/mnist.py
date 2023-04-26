import torch

from e2cnn import gspaces
from e2cnn import nn
from tqdm import tqdm
from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.datasets import MNIST
import sys
import numpy as np

from PIL import Image

class MnistRotDataset(Dataset):
    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test']

        if mode == "train":
            file = "mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = "mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"

        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')

        self.data = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.targets = data[:, -1].astype(np.int64)
        self.num_samples = len(self.targets)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.targets)

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            # 28x28
            torch.nn.Conv2d(1, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            # 14x14
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            # 6x6
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            # 3x3
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=0),
            torch.nn.ReLU(),
            # 1x1
            torch.nn.Conv2d(256, 10, kernel_size=1, padding=0),
        )
    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = self.conv(input).reshape(input.shape[0], -1)
        return x

class TriCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.group = gspaces.TrivialOnR2()
        # self.conv = torch.nn.Sequential(
        #     # 28x28
        #     nn.R2Conv(nn.FieldType(self.group, 1 * [self.group.trivial_repr]),
        #               nn.FieldType(self.group, 32 * [self.group.regular_repr]),
        #               kernel_size=3, padding=1),
        #     nn.ReLU(nn.FieldType(self.group, 32 * [self.group.regular_repr])),
        #     nn.PointwiseMaxPool(nn.FieldType(self.group, 32 * [self.group.regular_repr]), 2),
        #     # 14x14
        #     nn.R2Conv(nn.FieldType(self.group, 32 * [self.group.regular_repr]),
        #               nn.FieldType(self.group, 64 * [self.group.regular_repr]),
        #               kernel_size=3, padding=0),
        #     nn.ReLU(nn.FieldType(self.group, 64 * [self.group.regular_repr])),
        #     nn.PointwiseMaxPool(nn.FieldType(self.group, 64 * [self.group.regular_repr]), 2),
        #     # 6x6
        #     nn.R2Conv(nn.FieldType(self.group, 64 * [self.group.regular_repr]),
        #               nn.FieldType(self.group, 128 * [self.group.regular_repr]),
        #               kernel_size=3, padding=1),
        #     nn.ReLU(nn.FieldType(self.group, 128 * [self.group.regular_repr])),
        #     nn.PointwiseMaxPool(nn.FieldType(self.group, 128 * [self.group.regular_repr]), 2),
        #     # 3x3
        #     nn.R2Conv(nn.FieldType(self.group, 128 * [self.group.regular_repr]),
        #               nn.FieldType(self.group, 256 * [self.group.regular_repr]),
        #               kernel_size=3, padding=0),
        #     nn.ReLU(nn.FieldType(self.group, 256 * [self.group.regular_repr])),
        #     # 1x1
        #     nn.R2Conv(nn.FieldType(self.group, 256 * [self.group.regular_repr]),
        #               nn.FieldType(self.group, 10 * [self.group.trivial_repr]),
        #               kernel_size=1, padding=0),
        # )
        self.conv = torch.nn.Sequential(
            # 28x28
            nn.R2Conv(nn.FieldType(self.group, 1 * [self.group.trivial_repr]),
                      nn.FieldType(self.group, 16 * [self.group.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.group, 16 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 16 * [self.group.regular_repr]), 2),
            # 14x14
            nn.R2Conv(nn.FieldType(self.group, 16 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 32 * [self.group.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.group, 32 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 32 * [self.group.regular_repr]), 2),
            # 6x6
            nn.R2Conv(nn.FieldType(self.group, 32 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 64 * [self.group.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.group, 64 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 64 * [self.group.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.group, 64 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 128 * [self.group.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.group, 128 * [self.group.regular_repr])),
            # 1x1
            nn.R2Conv(nn.FieldType(self.group, 128 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 10 * [self.group.trivial_repr]),
                      kernel_size=1, padding=0),
        )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, nn.FieldType(self.group, 1 * [self.group.trivial_repr]))
        x = self.conv(x).tensor.reshape(x.shape[0], -1)
        return x

class C2CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.group = gspaces.Rot2dOnR2(2)
        self.conv = torch.nn.Sequential(
            # 28x28
            nn.R2Conv(nn.FieldType(self.group, 1 * [self.group.trivial_repr]),
                      nn.FieldType(self.group, 32 * [self.group.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.group, 32 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 32 * [self.group.regular_repr]), 2),
            # 14x14
            nn.R2Conv(nn.FieldType(self.group, 32 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 64 * [self.group.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.group, 64 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 64 * [self.group.regular_repr]), 2),
            # 6x6
            nn.R2Conv(nn.FieldType(self.group, 64 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 128 * [self.group.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.group, 128 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 128 * [self.group.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.group, 128 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 256 * [self.group.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.group, 256 * [self.group.regular_repr])),
            # 1x1
            nn.R2Conv(nn.FieldType(self.group, 256 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 10 * [self.group.trivial_repr]),
                      kernel_size=1, padding=0),
        )
    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, nn.FieldType(self.group, 1 * [self.group.trivial_repr]))
        x = self.conv(x).tensor.reshape(x.shape[0], -1)
        return x

class C4CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.group = gspaces.Rot2dOnR2(4)
        self.conv = torch.nn.Sequential(
            # 28x28
            nn.R2Conv(nn.FieldType(self.group, 1 * [self.group.trivial_repr]),
                      nn.FieldType(self.group, 16 * [self.group.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.group, 16 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 16 * [self.group.regular_repr]), 2),
            # 14x14
            nn.R2Conv(nn.FieldType(self.group, 16 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 32 * [self.group.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.group, 32 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 32 * [self.group.regular_repr]), 2),
            # 6x6
            nn.R2Conv(nn.FieldType(self.group, 32 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 128 * [self.group.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.group, 128 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 128 * [self.group.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.group, 128 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 128 * [self.group.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.group, 128 * [self.group.regular_repr])),
            # 1x1
            nn.R2Conv(nn.FieldType(self.group, 128 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 10 * [self.group.trivial_repr]),
                      kernel_size=1, padding=0),
        )
    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, nn.FieldType(self.group, 1 * [self.group.trivial_repr]))
        x = self.conv(x).tensor.reshape(x.shape[0], -1)
        return x

class D4CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.group = gspaces.FlipRot2dOnR2(4)
        self.conv = torch.nn.Sequential(
            # 28x28
            nn.R2Conv(nn.FieldType(self.group, 1 * [self.group.trivial_repr]),
                      nn.FieldType(self.group, 16 * [self.group.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.group, 16 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 16 * [self.group.regular_repr]), 2),
            # 14x14
            nn.R2Conv(nn.FieldType(self.group, 16 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 32 * [self.group.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.group, 32 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 32 * [self.group.regular_repr]), 2),
            # 6x6
            nn.R2Conv(nn.FieldType(self.group, 32 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 64 * [self.group.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.group, 64 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 64 * [self.group.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.group, 64 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 128 * [self.group.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.group, 128 * [self.group.regular_repr])),
            # 1x1
            nn.R2Conv(nn.FieldType(self.group, 128 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 10 * [self.group.trivial_repr]),
                      kernel_size=1, padding=0),
        )
    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, nn.FieldType(self.group, 1 * [self.group.trivial_repr]))
        x = self.conv(x).tensor.reshape(x.shape[0], -1)
        return x

class C8CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.group = gspaces.Rot2dOnR2(8)
        self.conv = torch.nn.Sequential(
            # 28x28
            nn.R2Conv(nn.FieldType(self.group, 1 * [self.group.trivial_repr]),
                      nn.FieldType(self.group, 16 * [self.group.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.group, 16 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 16 * [self.group.regular_repr]), 2),
            # 14x14
            nn.R2Conv(nn.FieldType(self.group, 16 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 32 * [self.group.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.group, 32 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 32 * [self.group.regular_repr]), 2),
            # 6x6
            nn.R2Conv(nn.FieldType(self.group, 32 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 64 * [self.group.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.group, 64 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 64 * [self.group.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.group, 64 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 128 * [self.group.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.group, 128 * [self.group.regular_repr])),
            # 1x1
            nn.R2Conv(nn.FieldType(self.group, 128 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 10 * [self.group.trivial_repr]),
                      kernel_size=1, padding=0),
        )
    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, nn.FieldType(self.group, 1 * [self.group.trivial_repr]))
        x = self.conv(x).tensor.reshape(x.shape[0], -1)
        return x

class D1CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.group = gspaces.Flip2dOnR2(0.)
        self.conv = torch.nn.Sequential(
            # 28x28
            nn.R2Conv(nn.FieldType(self.group, 1 * [self.group.trivial_repr]),
                      nn.FieldType(self.group, 32 * [self.group.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.group, 32 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 32 * [self.group.regular_repr]), 2),
            # 14x14
            nn.R2Conv(nn.FieldType(self.group, 32 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 64 * [self.group.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.group, 64 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 64 * [self.group.regular_repr]), 2),
            # 6x6
            nn.R2Conv(nn.FieldType(self.group, 64 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 128 * [self.group.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.group, 128 * [self.group.regular_repr])),
            nn.PointwiseMaxPool(nn.FieldType(self.group, 128 * [self.group.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.group, 128 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 256 * [self.group.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.group, 256 * [self.group.regular_repr])),
            # 1x1
            nn.R2Conv(nn.FieldType(self.group, 256 * [self.group.regular_repr]),
                      nn.FieldType(self.group, 10 * [self.group.trivial_repr]),
                      kernel_size=1, padding=0),
        )
    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, nn.FieldType(self.group, 1 * [self.group.trivial_repr]))
        x = self.conv(x).tensor.reshape(x.shape[0], -1)
        return x

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    train_size = 50000
    test_size = 50000
    max_epochs = 50
    max_no_improve = 10

    totensor = ToTensor()

    # model = C4CNN().to(device)
    # model = C8CNN().to(device)
    model = C2CNN().to(device)
    # model = TriCNN().to(device)
    # model = D4CNN().to(device)
    # model = CNN().to(device)
    # model = D1CNN().to(device)

    mnist_train = MnistRotDataset(mode='train', transform=totensor)
    mnist_test = MnistRotDataset(mode='test', transform=totensor)
    # mnist_train = MNIST692547(root='mnist', train=True, download=True, transform=totensor)
    # mnist_test = MNIST692547(root='mnist', train=False, download=True, transform=totensor)
    # mnist_train = MNIST(root='mnist', train=True, download=True, transform=totensor)
    # mnist_test = MNIST(root='mnist', train=False, download=True, transform=totensor)

    subsample_train_indices = torch.randperm(len(mnist_train.data))[:train_size]
    subsample_test_indices = torch.randperm(len(mnist_test.data))[:test_size]

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=256, sampler=SubsetRandomSampler(subsample_train_indices))
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=2048, sampler=SubsetRandomSampler(subsample_test_indices))

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

    no_improve = 0
    best_acc = 0
    best_per_acc = [0 for _ in range(10)]
    pbar = tqdm(total=max_epochs)
    for epoch in range(max_epochs):
        model.train()
        for i, (x, t) in enumerate(train_loader):
            optimizer.zero_grad()

            x = x.to(device)
            t = t.to(device)

            y = model(x)

            loss = loss_function(y, t)

            loss.backward()

            optimizer.step()

        pbar.update()

        # total = 0
        # correct = 0
        # per_total = [0 for _ in range(10)]
        # per_correct = [0 for _ in range(10)]
        # with torch.no_grad():
        #     model.eval()
        #     for i, (x, t) in enumerate(test_loader):
        #         x = x.to(device)
        #         t = t.to(device)
        #
        #         y = model(x)
        #
        #         _, prediction = torch.max(y.data, 1)
        #         total += t.shape[0]
        #         correct += (prediction == t).sum().item()
        #
        #         for j in range(10):
        #             per_total[j] += (t == j).sum().item()
        #             per_correct[j] += (prediction == t)[t == j].sum().item()
        #
        #     per_acc = (np.array(per_correct) / np.array(per_total) * 100.).round(2).tolist()
        #     per_acc_dic = {}
        #     for j in range(10):
        #         per_acc_dic[j] = per_acc[j]
        #
        #     acc = np.round(correct / total * 100., 2)
        #     if acc > best_acc:
        #         best_acc = acc
        #         best_per_acc = per_acc_dic
        #         no_improve = 0
        #     else:
        #         no_improve += 1
        #         if no_improve >= max_no_improve:
        #             break
        #     pbar.set_description('epoch {} | no imp: {}, test: {}, best: {}, digit : {}'.
        #                          format(epoch, no_improve, acc, best_acc, best_per_acc))
    total = 0
    correct = 0
    per_total = [0 for _ in range(10)]
    per_correct = [0 for _ in range(10)]
    pred_count = np.zeros((10, 10))
    with torch.no_grad():
        model.eval()
        for i, (x, t) in enumerate(test_loader):
            x = x.to(device)
            t = t.to(device)

            y = model(x)

            _, prediction = torch.max(y.data, 1)
            total += t.shape[0]
            correct += (prediction == t).sum().item()

            for j in range(10):
                per_total[j] += (t == j).sum().item()
                per_correct[j] += (prediction == t)[t == j].sum().item()
                pred_count[j] += prediction[t == j].bincount(minlength=10).cpu().numpy()

        per_acc = (np.array(per_correct) / np.array(per_total) * 100.).round(2).tolist()
        per_acc_dic = {}
        for j in range(10):
            per_acc_dic[j] = per_acc[j]

        acc = np.round(correct / total * 100., 2)

        best_acc = acc
        best_per_acc = per_acc_dic
        pbar.set_description('test: {}, digit : {}'.
                             format(best_acc, best_per_acc))
    pbar.close()
    return best_acc, best_per_acc, pred_count

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