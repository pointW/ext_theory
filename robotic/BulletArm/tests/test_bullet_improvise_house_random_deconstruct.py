import unittest
import time
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from bulletarm import env_factory

class TestBulletImproviseHouseRandomDeconstruct(unittest.TestCase):
  env_config = {}

  planner_config = {'random_orientation': True}

  def testPlanner(self):
    self.env_config['render'] = True
    self.env_config['seed'] = 1
    env = env_factory.createEnvs(1,  'improvise_house_building_random_deconstruct', self.env_config, self.planner_config)
    env.reset()
    for i in range(7, -1, -1):
      action = env.getNextAction()
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(dones, 1)
    env.close()

