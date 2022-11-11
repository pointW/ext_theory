import copy
import os
import time
import datetime
import matplotlib.pyplot as plt
from more_itertools import windowed
import dill as pickle
import json
from tqdm import tqdm
import numpy as np
from parameters import args

# Transition object

import joblib

class Logger(object):
    '''
    Logger for train/test runs.

    Args:
      - log_dir: Directory to write log
      - num_envs: Number of environments running concurrently
    '''

    def __init__(self, log_dir, model, env='swiss_roll', log_dir_sub=None):
        # Logging variables
        self.env = env
        self.model = model

        # Create directory in the logging directory
        timestamp = time.time()
        timestamp = datetime.datetime.fromtimestamp(timestamp)
        if not log_dir_sub:
            self.base_dir = os.path.join(log_dir, '{}_{}_{}'.format(self.model, self.env, timestamp.strftime('%Y-%m-%d.%H:%M:%S')))
        else:
            self.base_dir = os.path.join(log_dir, log_dir_sub)
        self.base_dir += 'seed{}'.format(args.seed)
        print('Creating logging session at: {}'.format(self.base_dir))

        # Create subdirs to save important run info
        self.info_dir = os.path.join(self.base_dir, 'info')
        self.depth_heightmaps_dir = os.path.join(self.base_dir, 'depth_heightmaps')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.trans_dir = os.path.join(self.base_dir, 'transitions')
        self.checkpoint_dir = os.path.join(self.base_dir, 'checkpoint')

        os.makedirs(self.info_dir)
        os.makedirs(self.depth_heightmaps_dir)
        os.makedirs(self.models_dir)
        os.makedirs(self.trans_dir)
        os.makedirs(self.checkpoint_dir)

        self.model_losses = list()
        self.model_holdout_losses = list()

    def saveModelLossCurve(self, n=100):
        plt.style.use('ggplot')
        losses = np.array(self.model_losses)
        if len(losses) < n:
            return
        if len(losses.shape) == 1:
            losses = np.expand_dims(losses, 0)
        else:
            losses = np.moveaxis(losses, 1, 0)
        for loss in losses:
            plt.plot(np.mean(list(windowed(loss, n)), axis=1))

        plt.savefig(os.path.join(self.info_dir, 'model_loss_curve.pdf'))
        plt.yscale('log')
        plt.savefig(os.path.join(self.info_dir, 'model_loss_curve_log.pdf'))

        plt.close()

    def saveModelHoldoutLossCurve(self):
        plt.style.use('ggplot')
        losses = np.array(self.model_holdout_losses)
        if len(losses) < 1:
            return
        if len(losses.shape) == 1:
            losses = np.expand_dims(losses, 0)
        else:
            losses = np.moveaxis(losses, 1, 0)
        for loss in losses:
            plt.plot(loss)

        plt.savefig(os.path.join(self.info_dir, 'model_holdout_loss_curve.pdf'))
        plt.yscale('log')
        plt.savefig(os.path.join(self.info_dir, 'model_holdout_loss_curve_log.pdf'))

        plt.close()

    def saveModelLosses(self):
        np.save(os.path.join(self.info_dir, 'model_losses.npy'), self.model_losses)
        np.save(os.path.join(self.info_dir, 'model_holdout_losses.npy'), self.model_holdout_losses)
