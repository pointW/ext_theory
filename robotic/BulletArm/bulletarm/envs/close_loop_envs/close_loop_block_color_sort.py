import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations
# from bulletarm.planners.close_loop_house_building_1_planner import CloseLoopHouseBuilding1Planner
from bulletarm.planners.close_loop_block_arranging_planner import CloseLoopBlockArrangingPlanner
from bulletarm.pybullet.utils.constants import NoValidPositionException

class CloseLoopBlockColorSortEnv(CloseLoopEnv):
  '''Open loop block arranging task.

  The robot needs to stack all N cubic blocks. The number of blocks N is set by the config.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    # env specific parameters
    super().__init__(config)
    self.bin_num = config['bin_num']
    self.bin_dist = config['bin_dist']
    self.goal_pos_cube = None
    self.goal_pos_tri = None
    # self.workspace_size = self.workspace[0,1]-self.workspace[0,0]

    if self.bin_num > 0:
      bins = np.arange(0,255,255/self.bin_num)
      bins_end = np.arange(0,255,255/self.bin_num) + 255/self.bin_num
      self.ranges = np.stack((bins,bins_end),axis=0).transpose(1,0)

      even_num = [2*i for i in range(self.bin_num//2)]
      odd_num = [2*i+1 for i in range(self.bin_num//2)]
      self.left_range = self.ranges[even_num]
      self.right_range = self.ranges[odd_num]





  def createMat(self):
    ws_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[self.workspace_size/2, self.workspace_size/4, 0.002], rgbaColor=[0.2, 0.2, 0.2, 1])
    ws_id1 = pb.createMultiBody(baseMass=0,
                                baseVisualShapeIndex=ws_visual,
                                basePosition=[self.workspace[0].mean(), self.workspace[1][0]+self.workspace_size/4, 0],
                                baseOrientation=[0, 0, 0, 1])
    ws_id2 = pb.createMultiBody(baseMass=0,
                                baseVisualShapeIndex=ws_visual,
                                basePosition=[self.workspace[0].mean(), self.workspace[1][1]-self.workspace_size/4, 0],
                                baseOrientation=[0, 0, 0, 1])
    self.ws_id = [ws_id1, ws_id2]

  def randSampleGray(self, bin_num=10, dist='uniform'):
    
    bins = np.arange(0,255,255/bin_num)

    even_num = [2*i for i in range(bin_num//2)]
    odd_num = [2*i+1 for i in range(bin_num//2)]
    even_idx = np.random.choice(even_num)
    odd_idx = np.random.choice(odd_num)
    # np.digitize(mylist,bins)
    if dist == 'uniform':
      left_gray = bins[even_idx]
      left_gray = (np.random.random()*255/bin_num+left_gray)/255
      right_gray = bins[odd_idx]
      right_gray = (np.random.random()*255/bin_num+right_gray)/255
    elif dist == 'uniform_discrete':
      left_gray = bins[even_idx]/255
      # left_gray = (np.random.random()*255/bin_num+left_gray)/255
      right_gray = bins[odd_idx]/255
      # right_gray = (np.random.random()*255/bin_num+right_gray)/255
    elif dist == 'gaussian_incorrect':
      sigma = 1/bin_num*1.5
      left_gray = bins[even_idx]/255
      right_gray = bins[odd_idx]/255
      left_gray = np.clip(np.random.normal(left_gray, sigma), 0, 1)
      right_gray = np.clip(np.random.normal(right_gray, sigma), 0, 1)
    elif dist == 'gaussian_clip':
      sigma = 1/bin_num*1.5
      left_gray = bins[even_idx]/255+255/bin_num/2
      right_gray = bins[odd_idx]/255+255/bin_num/2
      left_gray = left_gray + np.clip(np.random.normal(0, sigma), -255/bin_num/2, 255/bin_num/2)
      right_gray = right_gray + np.clip(np.random.normal(0, sigma), -255/bin_num/2, 255/bin_num/2)

      left_gray = np.clip(left_gray, 0, 1)
      right_gray = np.clip(right_gray, 0, 1)
    elif dist == 'gaussian_entire':
      sigma = 255/2*0.8
      left_gray = None
      right_gray = None

      while (left_gray == None) or (right_gray == None):
        samp = np.random.normal(255/2, sigma)
        for count, current_range in enumerate(self.ranges):
          if current_range[0] <= samp < current_range[1]:
            if count % 2 == 0:
              left_gray = samp
            else:
              right_gray = samp

      left_gray = np.clip(left_gray/255, 0, 1)
      right_gray = np.clip(right_gray/255, 0, 1)


    # left_red = bins[odd_idx]
    # left_red = (np.random.random()*255/bin_num+left_red)/255
    # right_red = bins[even_idx]
    # right_red = (np.random.random()*255/bin_num+right_red)/255

    # three_num = [3*i for i in range(bin_num//3)]
    # threeone_num = [3*i+1 for i in range(bin_num//3)]
    # three_idx = np.random.choice(three_num)
    # threeone_idx = np.random.choice(threeone_num)
    # left_green = bins[three_idx]
    # left_green = (np.random.random()*255/bin_num+left_green)/255
    # right_green = bins[threeone_idx]
    # right_green = (np.random.random()*255/bin_num+right_green)/255

    left_random = np.random.random()
    right_random = np.random.random()

    left_rgba = [left_gray, left_gray, left_gray, 1]
    right_rgba = [right_gray, right_gray, right_gray, 1]
    return left_rgba, right_rgba

  def randomCentralPos(self, area):
    # area is a 3x2 matrix in the same form of self.workspace
    pos = []
    for i in range(3):
      area_size = abs(area[i][1]-area[i][0])
      coor = np.random.random()*area_size+area[i][0]
      pos.append(coor)
    return pos

  def reset(self):

    while True:
      self.resetPybulletWorkspace()
      try:
        # pos = self.randomCentralPos(self.area)
        self.handle_cube = self._generateShapes(constants.CUBE_BIG, 1, scale=1, random_orientation=self.random_orientation)
        self.handle_triangle = self._generateShapes(constants.TRIANGLE_BIG, 1, scale=1, random_orientation=self.random_orientation)
        # generate yellow blocks
        pb.changeVisualShape(self.handle_cube[0].object_id, -1, rgbaColor=[1, 1, 0, 1])
        pb.changeVisualShape(self.handle_triangle[0].object_id, -1, rgbaColor=[1, 1, 0, 1])

        if self.custom_workspace:
          # change workspace color
          # print(f"bin_num:{self.bin_num},bin_dist:{self.bin_dist}")
          left_rgba, right_rgba = self.randSampleGray(self.bin_num, dist=self.bin_dist)
          # print(left_rgba, right_rgba)
          pb.changeVisualShape(self.ws_id[0], -1, rgbaColor=left_rgba)
          pb.changeVisualShape(self.ws_id[1], -1, rgbaColor=right_rgba)

        # left side and right side sorting task
        self.goal_pos_cube = [0.45, -0.09]
        self.goal_pos_tri = [0.45, 0.09]
      except NoValidPositionException as e:
        continue
      else:
        break

    return self._getObservation()


  def _checkTermination(self):
    ''''''
    obj_pos_cube = self.handle_cube[0].getPosition()[:2]
    obj_pos_tri = self.handle_triangle[0].getPosition()[:2]
    sort_cube = abs(obj_pos_cube[1] - self.goal_pos_cube[1]) < 0.02
    sort_tri = abs(obj_pos_tri[1] - self.goal_pos_tri[1]) < 0.02
    return sort_cube and sort_tri


  def isSimValid(self):
    for obj in self.objects:
      p = obj.getPosition()
      if self._isObjectHeld(obj):
        continue
      if not self.workspace[0][0]-0.05 < p[0] < self.workspace[0][1]+0.05 and \
          self.workspace[1][0]-0.05 < p[1] < self.workspace[1][1]+0.05 and \
          self.workspace[2][0] < p[2] < self.workspace[2][1] :
        return False
    return True


def createCloseLoopBlockColorSortEnv(config):
  return CloseLoopBlockColorSortEnv(config)


if __name__ == '__main__':
  env = CloseLoopBlockColorSortEnv({'seed': 0, 'render': True, 'workspace_option': 'white_plane,custom,trans_robot,random_init_gripper', 'view_type': 'camera_center_xyz_rgbd_noGripper', 'robot':'panda', 'bin_dist':'gaussian_entire', 'bin_num':10})


  planner = CloseLoopBlockArrangingPlanner(env, {})
  (state, in_hands, obs) = env.reset()

  while True:
    action = planner.getNextAction()
    (state, in_hands, obs), reward, done = env.step(action)

    if done:
      # print('done')
      (state, in_hands, obs) = env.reset()
      print(1)