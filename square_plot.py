import copy
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import ternary

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def plotLoss(base, step, name='model_holdout_losses'):
    colors = "bgrycmkw"
    method_map = {
        'mlp': 'MLP',
        'dssz': 'INV',
        'invz': 'INV',
    }
    i = 0

    data = np.zeros((6, 5))
    for method in sorted(get_immediate_subdirectories(base)):
        rs = []
        test_loss = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/{}.npy'.format(name)))
                rs.append(r[:, 0])
                # test_loss.append(r.min())
                test_loss.append(r[r[:, 0].argmin(), 1])
                # test_loss.append(np.sort(r)[:5].mean())
            except Exception as e:
                continue
        assert j == 9

        print('{}, ${:.3f} \pm {:.3f}$'
              .format(method,
                      np.mean(test_loss), stats.sem(test_loss),
                      ))

        _, model, ndata, mr, cr = method.split('_')
        mr = float(mr.removeprefix('mr'))
        cr = float(cr.removeprefix('cr'))
        ub = (1-mr) * (1-cr)
        mr = round(mr / 0.2) - 1
        cr = round(cr / 0.2)
        data[cr, mr] = np.abs(1 - np.mean(test_loss) - ub)
        if cr == 5:
            data[cr, :] = np.abs(1 - np.mean(test_loss) - ub)

    plt.figure(dpi=300)

    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    plt.imshow(data.T*100)
    plt.yticks([0, 1, 2, 3, 4], [0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks([0, 1, 2, 3, 4, 5], [0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar = plt.colorbar()
    cbar.set_label('Percentage')
    plt.title('L1 Distance Between Error and LB (%)')
    plt.ylabel('m')
    plt.xlabel('c')
    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(base, 'square.png'), bbox_inches='tight', pad_inches=0)
    print(1)

        # print('${:.3f} \pm {:.3f}$'
        #       .format(np.mean(test_loss), stats.sem(test_loss),
        #               ))


if __name__ == '__main__':
    # base = '/media/dian/hdd/mrun_results/transfer/0822_topdown'
    # for sub in os.listdir(base):
    #     if not os.path.isdir(os.path.join(base, sub)):
    #         continue
    #     plotEvalCurve(os.path.join(base, sub), 10000, freq=500)

    base = '/media/dian/hdd/mrun_results/swiss_roll/square'
    plotLoss(base, 10000)
