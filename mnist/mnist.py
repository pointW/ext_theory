import torch

from e2cnn import gspaces
from e2cnn import nn
from tqdm import tqdm

class C8SteerableCNN(torch.nn.Module):

    def __init__(self, n_classes=10):
        super(C8SteerableCNN, self).__init__()

        # the model is equivariant under rotations by 45 degrees, modelled by C8
        # self.r2_act = gspaces.Rot2dOnR2(N=2)
        self.r2_act = gspaces.TrivialOnR2()

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C8
        out_type = nn.FieldType(self.r2_act, 24 * [self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.MaskModule(in_type, 29, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96 * [self.r2_act.regular_repr])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96 * [self.r2_act.regular_repr])
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr])
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        self.gpool = nn.GroupPooling(out_type)

        # number of output channels
        c = self.gpool.out_type.size

        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, n_classes),
        )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.input_type)

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)

        x = self.block5(x)
        x = self.block6(x)

        # pool over the spatial dimensions
        x = self.pool3(x)

        # pool over the group
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x

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

from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.datasets import MNIST

import numpy as np

from PIL import Image

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


class MnistRotDataset(Dataset):

    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test']

        if mode == "train":
            file = "mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = "mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"

        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')

        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        mask = (self.labels == 6) + (self.labels == 9)
        self.images = self.images[mask]
        self.labels = self.labels[mask]
        self.num_samples = len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)

train_size = 5000
test_size = 1000

# class MNIST69(MNIST):
#     def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
#         super().__init__(root, train, transform, target_transform, download)
#         mask = (self.targets == 6) + (self.targets == 9)
#         self.data = self.data[mask]
#         self.targets = self.targets[mask]
#         if train:
#             assert self.data.shape[0] > train_size
#             self.data = self.data[:train_size]
#             self.targets = self.targets[:train_size]
#         else:
#             assert self.data.shape[0] > test_size
#             self.data = self.data[:test_size]
#             self.targets = self.targets[:test_size]
#
# class MNIST692547(MNIST):
#     def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
#         super().__init__(root, train, transform, target_transform, download)
#         mask = (self.targets == 6) + (self.targets == 9) + \
#                (self.targets == 2) + (self.targets == 5) + \
#                (self.targets == 4) + (self.targets == 7)
#         self.data = self.data[mask]
#         self.targets = self.targets[mask]
#         if train:
#             assert self.data.shape[0] > train_size
#             self.data = self.data[:train_size]
#             self.targets = self.targets[:train_size]
#         else:
#             assert self.data.shape[0] > test_size
#             self.data = self.data[:test_size]
#             self.targets = self.targets[:test_size]

# build the test set
# raw_mnist_test = MnistRotDataset(mode='test')
# raw_mnist_test = MNIST69(root='mnist', train=False, download=True)

# images are padded to have shape 29x29.
# this allows to use odd-size filters with stride 2 when downsampling a feature map in the model
pad = Pad((0, 0, 1, 1), fill=0)

# to reduce interpolation artifacts (e.g. when testing the model on rotated images),
# we upsample an image by a factor of 3, rotate it and finally downsample it again
resize1 = Resize(87)
resize2 = Resize(29)

totensor = ToTensor()

# model = C4CNN().to(device)
# model = C2CNN().to(device)
# model = TriCNN().to(device)
model = D4CNN().to(device)
# model = CNN().to(device)

train_transform = Compose([
    pad,
    resize1,
    RandomRotation(180, resample=Image.BILINEAR, expand=False),
    resize2,
    totensor,
])


test_transform = Compose([
    pad,
    totensor,
])

# mnist_train = MnistRotDataset(mode='train', transform=train_transform)
# mnist_test = MnistRotDataset(mode='test', transform=test_transform)
# mnist_train = MNIST692547(root='mnist', train=True, download=True, transform=totensor)
# mnist_test = MNIST692547(root='mnist', train=False, download=True, transform=totensor)
mnist_train = MNIST(root='mnist', train=True, download=True, transform=totensor)
mnist_test = MNIST(root='mnist', train=False, download=True, transform=totensor)

subsample_train_indices = torch.randperm(len(mnist_train.data))[:train_size]
subsample_test_indices = torch.randperm(len(mnist_test.data))[:test_size]

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, sampler=SubsetRandomSampler(subsample_train_indices))
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=2048, sampler=SubsetRandomSampler(subsample_test_indices))

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

max_epochs = 1000
max_no_improve = 50
no_improve = 0
# pbar = tqdm(total=max_epochs)
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

    total = 0
    correct = 0
    per_total = [0 for _ in range(10)]
    per_correct = [0 for _ in range(10)]
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

        per_acc = (np.array(per_correct) / np.array(per_total) * 100.).round(2).tolist()
        per_acc_dic = {}
        for j in range(10):
            per_acc_dic[j] = per_acc[j]

        acc = np.round(correct / total * 100., 2)
        if acc > best_acc:
            best_acc = acc
            best_per_acc = per_acc_dic
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
        pbar.set_description('epoch {} | no imp: {}, test: {}, best: {}, digit : {}'.
                             format(epoch, no_improve, acc, best_acc, best_per_acc))
        pbar.update()
