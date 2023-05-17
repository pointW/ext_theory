import unittest
import time
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from bulletarm import env_factory

class TestBulletCloseLoopBlockArranging(unittest.TestCase):
  env_config = {'planner_mix_stack': 2, 'planner_mix_sort': 2, 'view_type':'camera_center_xyz_rgbd_noGripper',
                 'workspace_option':'trans_robot,white_plane', 'robot':'panda'}

  planner_config = {'random_orientation': True, 'dpos': 0.05, 'drot': np.pi / 4}

  def testPlanner2(self):
    self.env_config['render'] = True
    self.env_config['seed'] = 0
    num_processes = 1
    env = env_factory.createEnvs(num_processes, 'close_loop_block_arranging', self.env_config, self.planner_config)
    task_stack = 0
    task_sort = 0
    s = 0
    step_times = []

    flags = [True for i in range(num_processes)]
    env.setFlag(flags=flags)
    (states, in_hands, obs) = env.reset()
    total_stack = self.env_config['planner_mix_stack']
    pbar = tqdm(total=total_stack)

    while task_stack < total_stack:
      # env.setFlag(flag=True)
      t0 = time.time()
      action = env.getNextAction()
      t_plan = time.time() - t0
      # plt.imshow(obs[0, 0])
      # plt.show()

      # (states_, in_hands_, obs_), rewards, dones = env.simulate(action)
      # plt.imshow(obs_[0, 0])
      # plt.show()

      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset= True)
      # plt.imshow(np.transpose(obs_[0, 0:3], (1, 2, 0)))
      # plt.show()
      # if rewards:
      #   print(1)
      obs = obs_
      s += rewards.sum()
      task_stack += dones.sum()
      t_action = time.time() - t0 - t_plan
      t = time.time() - t0
      step_times.append(t)

      pbar.set_description(
        '{:.3f}/{}, SR: {:.3f}, plan time: {:.2f}, action time: {:.2f}, avg step time: {:.2f}'
          .format(s, task_stack, float(s) / task_stack if task_stack != 0 else 0, t_plan, t_action, np.mean(step_times))
      )

    flags = [False for i in range(num_processes)]
    env.setFlag(flags=flags)
    (states, in_hands, obs) = env.reset()
    total_sort = self.env_config['planner_mix_sort']
    pbar = tqdm(total=total_sort)

    while task_sort < total_sort:
      t0 = time.time()
      # env.setPlannerFlag(flags=[False for i in range(num_processes)])
      action = env.getNextAction()
      t_plan = time.time() - t0
      # plt.imshow(obs[0, 0])
      # plt.show()

      # (states_, in_hands_, obs_), rewards, dones = env.simulate(action)
      # plt.imshow(obs_[0, 0])
      # plt.show()

      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset= True)
      # plt.imshow(obs_[0, 0])
      # plt.show()
      # if rewards:
      #   print(1)
      obs = obs_
      s += rewards.sum()
      task_sort += dones.sum()
      t_action = time.time() - t0 - t_plan
      t = time.time() - t0
      step_times.append(t)

      pbar.set_description(
        '{:.3f}/{}, SR: {:.3f}, plan time: {:.2f}, action time: {:.2f}, avg step time: {:.2f}'
          .format(s, task_sort, float(s) / task_sort if task_sort != 0 else 0, t_plan, t_action, np.mean(step_times))
      )
    env.close()

if __name__ == "__main__":
  TestBulletCloseLoopBlockArranging().testPlanner2()