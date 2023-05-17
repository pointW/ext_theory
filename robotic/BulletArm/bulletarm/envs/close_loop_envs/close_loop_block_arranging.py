import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations
from bulletarm.pybullet.utils.constants import NoValidPositionException

class CloseLoopBlockArrangingEnv(CloseLoopEnv):
  '''Open loop block arranging task.

  The robot needs to stack all N cubic blocks. The number of blocks N is set by the config.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    # env specific parameters
    if 'num_objects' not in config:
      config['num_objects'] = 2
    super().__init__(config)
    # self.arrange_stack = config['planner_mix_stack'] and True
    # self.arrange_sort = config['planner_mix_sort'] and True
    self.goal_pos_cube = self.workspace.mean(1)[:2]
    self.goal_pos_tri = self.workspace.mean(1)[:2]
    # if self.arrange_stack and self.arrange_sort and True:
    #   print('---------------incorrect environment---------------')
    #   self.arrange_sort = False
    self.arrange_stack = True

  def setFlag(self, flag):
    self.arrange_stack = flag

  def reset(self):
    while True:
      self.resetPybulletWorkspace()
      if self.arrange_stack:
        self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
      try:
        if self.arrange_stack:
          self.handle_triangle = self._generateShapes(constants.TRIANGLE, 1, scale=1, random_orientation=self.random_orientation)
          self.handle_cube = self._generateShapes(constants.CUBE, 1, scale=1, random_orientation=self.random_orientation)
          # generate different color for stack
          pb.changeVisualShape(self.handle_cube[0].object_id, -1, rgbaColor=[0, 0, 1, 1])
          pb.changeVisualShape(self.handle_triangle[0].object_id, -1, rgbaColor=[0, 1, 0, 1])
        else:
          self._generateShapes(constants.CUBE_BIG, 1, scale=1, random_orientation=self.random_orientation)
          self._generateShapes(constants.TRIANGLE_BIG, 1, scale=1, random_orientation=self.random_orientation)

          # left side and right side sorting task
          self.goal_pos_cube = [0.45, -0.09]
          self.goal_pos_tri = [0.45, 0.09]
      except NoValidPositionException as e:
        continue
      else:
        break

    return self._getObservation()


  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation


  def _checkTermination(self):
    ''''''
    if self.arrange_stack:
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
      return not self._isHolding() and self._checkStack(blocks + triangles) and self._checkObjUpright(triangles[0]) and self._isObjOnTop(triangles[0])

    else:
      obj_pos_cube = self.objects[0].getPosition()[:2]
      obj_pos_tri = self.objects[1].getPosition()[:2]
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
          self.workspace[2][0] < p[2] < self.workspace[2][1]:
        return False
    return True


def createCloseLoopBlockArrangingEnv(config):
  return CloseLoopBlockArrangingEnv(config)
