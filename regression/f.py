import numpy as np
import numpy.random as npr

n = int(1e6)
# npr.seed(0)
# F_type = 'equivariant'
F_type = 'random'
Inv_rot = [[[1, 0],
            [0, 1]],
           [[0, 1],
            [-1, 0]],
           [[-1, 0],
            [0, -1]],
           [[0, -1],
            [1, 0]]]
Inv_rot = np.asarray(Inv_rot)
order = 4


class Poly_F:
    def __init__(self):
        if F_type == 'random':
            # random parameters
            self.weight_x = npr.randn(4, order)
            self.weight_y = npr.randn(4, order)
        elif F_type in ['equivariant', 'invariant']:
            self.weight_x = npr.randn(1, order)
            self.weight_y = npr.randn(1, order)

    def LBfIR(self, axis=None, value=None):
        axis = np.arange(4).reshape(1, 4).repeat(n, axis=0).reshape(-1)
        value = npr.rand(n).reshape(n, 1).repeat(4, axis=1).reshape(-1)
        f = self.forward(axis, value).reshape(n, 4, 2)
        EGxf = f.mean(axis=1)[:, None, :]
        return ((f - EGxf) ** 2).mean()

    # def EG_invf(self):
    #     axis = npr.randint(0, 4, [n])
    #     value = npr.rand(n)
    #     f = self.forward(axis, value)  # n x 2
    #     G_inv = Inv_rot[axis]  # n x 2 x 2
    #     G_invf = np.matmul(G_inv, f[:, :, None]).squeeze(-1)  # n x 2
    #     return G_invf.mean(axis=0)

    def LBfER(self, axis=None, value=None):
        axis = np.arange(4).reshape(1, 4).repeat(n, axis=0).reshape(-1) if axis is None else axis
        value = npr.rand(n).reshape(n, 1).repeat(4, axis=1).reshape(-1) if value is None else value
        f = self.forward(axis, value).reshape(-1, 2)
        G_inv = Inv_rot[axis]  # 4n x 2 x 2
        G_invf = np.matmul(G_inv, f[:, :, None]).reshape(n, 4, 2)  # 4n x 2
        EGxG_invf = G_invf.mean(axis=1)[:, None, :].repeat(4, axis=1)  # n x 4 x 2
        EGxG_invf = EGxG_invf.reshape(-1, 2)  # 4n x 2
        axis_inv = -axis % 4
        g = Inv_rot[axis_inv]  # 4n x 2 x 2
        self.gEGxG_invf = np.matmul(g, EGxG_invf[:, :, None]).squeeze(-1)
        return ((f - self.gEGxG_invf) ** 2).mean()

    def matmul(self, values, axis):
        fx = np.matmul(values[:, None, :], self.weight_x[axis][:, :, None]).squeeze(-1)
        fy = np.matmul(values[:, None, :], self.weight_y[axis][:, :, None]).squeeze(-1)
        return fx, fy

    def forward(self, axis, value):
        assert ((value >= 0) & (value <= 1)).all()
        values = list()
        for power in range(order):
            values.append(value ** power)
        values = np.asarray(values).transpose([1, 0])
        if F_type == 'random':
            fx = np.matmul(values[:, None, :], self.weight_x[axis][:, :, None]).squeeze(-1)
            fy = np.matmul(values[:, None, :], self.weight_y[axis][:, :, None]).squeeze(-1)
            f = np.concatenate([fx, fy], axis=1)
        elif F_type == 'invariant':
            fx = np.matmul(values[:, None, :], self.weight_x[0][None, :, None]).squeeze(-1)
            fy = np.matmul(values[:, None, :], self.weight_y[0][None, :, None]).squeeze(-1)
            f = np.concatenate([fx, fy], axis=1)
        elif F_type == 'equivariant':
            fx = np.matmul(values[:, None, :], self.weight_x[0][None, :, None]).squeeze(-1)
            fy = np.matmul(values[:, None, :], self.weight_y[0][None, :, None]).squeeze(-1)
            ginv_f = np.concatenate([fx, fy], axis=1)  # n x 2
            axis_inv = -axis % 4
            g = Inv_rot[axis_inv]  # n x 2 x 2
            f = np.matmul(g, ginv_f[:, :, None]).squeeze(-1)
        else:
            raise NotImplementedError
        return f


if __name__ == '__main__':
    F = Poly_F()
    print(F.LBfIR())
    print(F.LBfER())
