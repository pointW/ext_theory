from utils.parameters import *
from networks.cnn import Actor

from agents.bc_continuous import BehaviorCloningContinuous

from networks.equivariant_sac_net import EquivariantPolicyFlip

def createAgent(test=False):
    print('initializing agent')
    if view_type == 'camera_fix_rgbd':
        obs_channel = 5
    elif view_type in ['camera_center_xyz_rgbd', 'camera_center_xyz_rgbd_noGripper']:
        obs_channel = 5
    else:
        obs_channel = 2
    if load_sub is not None or load_model_pre is not None or test:
        initialize = False
    else:
        initialize = True
    n_p = 2
    if not random_orientation:
        n_theta = 1
    else:
        n_theta = 3

    # setup agent
    if alg in ['bc_con']:
        agent = BehaviorCloningContinuous(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                                          n_a=len(action_sequence))

        if model == 'cnn':
            policy = Actor((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
        elif model == 'equi_d1':
            policy = EquivariantPolicyFlip((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(policy)


    else:
        raise NotImplementedError
    agent.aug = aug
    agent.aug_type = aug_type
    print('initialized agent')
    return agent