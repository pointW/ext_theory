import numpy as np
import torch.optim
import matplotlib.pyplot as plt
from f import *
from network import *

device = torch.device('cuda')


def generate_data(f, n_x, bias='random'):
    if bias == 'random':
        axis = npr.randint(0, 4, [n_x])
        value = npr.rand(n_x)
    elif bias == 'equivariant':
        axis = np.arange(4).reshape(1, 4).repeat(n_x // 4, axis=0).reshape(-1)
        value = npr.rand(n_x // 4).reshape(n_x // 4, 1).repeat(4, axis=1).reshape(-1)
    y = f.forward(axis, value)
    axis = torch.tensor(axis).reshape(-1, 1).float().to(device)
    value = torch.tensor(value).reshape(-1, 1).float().to(device)
    y = torch.tensor(y).reshape(-1, 2).float().to(device)
    return axis, value, y


def main(h_type, log=False):
    print('=========>    Evaluating ', h_type)
    max_iters = int(2e2)
    # h = Inv_h() if h_type == 'invariant' else C4_h()
    h = Inv_h() if h_type == 'invariant' else Equ_h()
    h.to(device)
    f = Poly_F()
    optimizer = torch.optim.Adam(h.parameters(), lr=1e-3, weight_decay=1e-5)
    bs = int(1e5)
    if log:
        theo_err_h = f.LBfIR() if h_type == 'invariant' else f.LBfER()
        print('Theoretical err(h): {:.4f}'.format(theo_err_h))

    # Train
    npr.seed(None)
    for iter in range(max_iters):
        axis, value, y = generate_data(f, bs)
        loss = nn.functional.mse_loss(y, h(axis, value))
        if iter > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if iter % (max_iters // 10) == 0 and log:
            print('Iteration: {}, loss: {:.3f}'.format(iter, loss))

    # Test
    npr.seed(0)
    with torch.no_grad():
        axis, value, y = generate_data(f, n, bias='equivariant')
        y_hat = h(axis, value)
        err_h = nn.functional.mse_loss(y, y_hat)
        theo_err_h = f.LBfIR(axis, value) if h_type == 'invariant' else f.LBfER()

    if log:
        print('Empirical err(h): {:.4f}'.format(err_h))
        print('Theoretical err(h): {:.4f}'.format(theo_err_h))
        print('Relation emp/theo: {:.4f}'.format(err_h/theo_err_h))

    # plt.figure()
    # y = y.cpu()
    # y_hat = y_hat.cpu()
    # theo_y = f.gEGxG_invf
    # plt.quiver(y[:8, 0], y[:8, 1], color=['r'])
    # plt.quiver(y_hat[:8, 0], y_hat[:8, 1], color=['g'])
    # plt.quiver(theo_y[:8, 0], theo_y[:8, 1], color=['b'])
    # plt.show()

    return (err_h/theo_err_h).item()


if __name__ == '__main__':
    i_relation, e_relation = list(), list()
    for n_run in range(100):
        i_relation.append(main('invariant'))
        e_relation.append(main('equivariant'))

    i_relation, e_relation = np.asarray(i_relation), np.asarray(e_relation)
    print('Invariant relation: {:.4f}+-{:.8f}'.format(i_relation.mean(), i_relation.var()))
    print('Equivariant relation: {:.4f}+-{:.8f}'.format(e_relation.mean(), e_relation.var()))
