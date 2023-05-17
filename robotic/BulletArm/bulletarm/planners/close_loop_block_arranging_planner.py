import numpy as np
import scipy
from bulletarm.planners.close_loop_planner import CloseLoopPlanner
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils import transformations
from bulletarm.envs.close_loop_envs.close_loop_block_arranging import CloseLoopBlockArrangingEnv


class CloseLoopBlockArrangingPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    # self.arrange_sort = env.arrange_sort and True

    # if self.arrange_stack:
    self.pick_place_stage = 0
    self.current_target = None
    self.previous_target = None
    self.target_obj = None
    # else:
    self.target_push = None
    self.stage = 0
    self.pre_push_start_pos_cube = self.env.workspace.mean(1)
    self.push_start_pos_cube = self.env.workspace.mean(1)
    self.push_end_pos_cube = self.env.workspace.mean(1)
    self.push_rot = 0
    self.pre_push_start_pos_tri = self.env.workspace.mean(1)
    self.push_start_pos_tri = self.env.workspace.mean(1)
    self.push_end_pos_tri = self.env.workspace.mean(1)
    self.push_rot_tri = 0

  def setPlannerFlag(self, flag):
    self.arrange_stack = flag

  def getNextActionToCurrentTarget(self):
    if self.arrange_stack:
      x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
      if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
        primitive = constants.PICK_PRIMATIVE if self.current_target[2] is constants.PICK_PRIMATIVE else constants.PLACE_PRIMATIVE
        self.previous_target = self.current_target
        self.current_target = None
      else:
        primitive = constants.PICK_PRIMATIVE if self.isHolding() else constants.PLACE_PRIMATIVE
      return self.env._encodeAction(primitive, x, y, z, r)
    else:
      x, y, z, r = self.getActionByGoalPose(self.target_push[0], self.target_push[1])
      p = self.target_push[2]
      if np.all(np.abs([x, y, z]) < self.dpos) and (not self.random_orientation or np.abs(r) < self.drot):
        self.target_push = None
      return self.env._encodeAction(p, x, y, z, r)

  def setWaypoints(self):
    obj_pos_cube = self.env.getObjectPositions(omit_hold=False)[:, :2].tolist()[0]
    obj_pos_tri = self.env.getObjectPositions(omit_hold=False)[:, :2].tolist()[1]
    goal_pos_cube = self.env.goal_pos_cube
    goal_pos_tri = self.env.goal_pos_tri

    d = 0.04
    p = [obj_pos_cube[0], obj_pos_cube[1] + d * 2]
    g = [obj_pos_cube[0], goal_pos_cube[1] + d / 2]
    p_tri = [obj_pos_tri[0], obj_pos_tri[1] - d * 2]
    g_tri = [obj_pos_tri[0], goal_pos_tri[1] - d / 2]

    p[0] = np.clip(p[0], self.env.workspace[0][0], self.env.workspace[0][1])
    p[1] = np.clip(p[1], self.env.workspace[1][0], self.env.workspace[1][1])
    self.pre_push_start_pos_cube = (p[0], p[1], self.env.workspace[2][0] + 0.1)
    self.push_start_pos_cube = (p[0], p[1], self.env.workspace[2][0]+0.02)
    self.push_end_pos_cube = (g[0], g[1], self.env.workspace[2][0]+0.02)
    self.push_pos_lift = (g[0], g[1], self.env.workspace[2][0] + 0.08)
    self.push_rot = 0
    p_tri[0] = np.clip(p_tri[0], self.env.workspace[0][0], self.env.workspace[0][1])
    p_tri[1] = np.clip(p_tri[1], self.env.workspace[1][0], self.env.workspace[1][1])
    self.pre_push_start_pos_tri = (p_tri[0], p_tri[1], self.env.workspace[2][0] + 0.1)
    self.push_start_pos_tri = (p_tri[0], p_tri[1], self.env.workspace[2][0] + 0.01)
    self.push_end_pos_tri = (g_tri[0], g_tri[1], self.env.workspace[2][0] + 0.01)
    self.push_rot_tri = 0

  def setNewTarget(self):
    if self.arrange_stack:
      blocks = np.array(list(filter(lambda x: x.object_type_id is constants.CUBE and not self.isObjectHeld(x) and self.isObjOnTop(x), self.env.objects)))
      if not blocks:
        blocks = np.array(list(filter(lambda x: x.object_type_id is constants.CUBE, self.env.objects)))
      block_poses = self.env.getObjectPoses(blocks)
      sorted_inds = np.flip(np.argsort(block_poses[:,2], axis=0))
      blocks = blocks[sorted_inds]

      triangle = self.env.objects[0]

      if self.env.current_episode_steps == 1:
        self.pick_place_stage = 0

      if self.pick_place_stage in [0, 1, 2]:
        if self.target_obj is None:
          self.target_obj = blocks[1] if len(blocks) > 1 else triangle
        object_pos = self.target_obj.getPosition()
        object_rot = list(transformations.euler_from_quaternion(self.target_obj.getRotation()))
        gripper_rz = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())[2]
        if self.target_obj.object_type_id == constants.TRIANGLE:
          while object_rot[2] - gripper_rz > np.pi/2:
            object_rot[2] -= np.pi
          while object_rot[2] - gripper_rz < -np.pi/2:
            object_rot[2] += np.pi
        else:
          while object_rot[2] - gripper_rz > np.pi/4:
            object_rot[2] -= np.pi/2
          while object_rot[2] - gripper_rz < -np.pi/4:
            object_rot[2] += np.pi/2
        pre_pick_pos = object_pos[0], object_pos[1], object_pos[2] + 0.1
        if self.pick_place_stage == 0:
          self.pick_place_stage = 1
          self.current_target = (pre_pick_pos, object_rot, constants.PLACE_PRIMATIVE)
        elif self.pick_place_stage == 1:
          self.pick_place_stage = 2
          self.current_target = (object_pos, object_rot, constants.PICK_PRIMATIVE)
        else:
          self.pick_place_stage = 3
          self.target_obj = None
          self.current_target = (pre_pick_pos, object_rot, constants.PICK_PRIMATIVE)

      else:
        if self.target_obj is None:
          self.target_obj = blocks[0]
        object_pos = self.target_obj.getPosition()
        object_rot = list(transformations.euler_from_quaternion(self.target_obj.getRotation()))
        gripper_rz = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())[2]
        if self.target_obj.object_type_id == constants.TRIANGLE:
          while object_rot[2] - gripper_rz > np.pi/2:
            object_rot[2] -= np.pi
          while object_rot[2] - gripper_rz < -np.pi/2:
            object_rot[2] += np.pi
        else:
          while object_rot[2] - gripper_rz > np.pi / 4:
            object_rot[2] -= np.pi / 2
          while object_rot[2] - gripper_rz < -np.pi / 4:
            object_rot[2] += np.pi / 2
        pre_place_pos = object_pos[0], object_pos[1], object_pos[2] + 0.1
        if self.pick_place_stage == 3:
          self.pick_place_stage = 4
          self.current_target = (pre_place_pos, object_rot, constants.PICK_PRIMATIVE)
        elif self.pick_place_stage == 4:
          self.pick_place_stage = 5
          place_pos = object_pos[0], object_pos[1], object_pos[2] + self.getMaxBlockSize() * 1.2
          self.current_target = (place_pos, object_rot, constants.PLACE_PRIMATIVE)
        else:
          self.pick_place_stage = 0
          self.target_obj = None
          self.current_target = (pre_place_pos, object_rot, constants.PLACE_PRIMATIVE)

    else:
      if self.stage == 0:
        self.setWaypoints()
        # to pre push start pos
        self.target_push = (self.pre_push_start_pos_cube, self.push_rot, 0.5, 0.5)
        self.stage = 1
      elif self.stage == 1:
        # to push start pos
        self.target_push = (self.push_start_pos_cube, self.push_rot, 0.5, 0.5)
        self.stage = 2
      elif self.stage == 2:
        # to push end pos
        self.target_push = (self.push_end_pos_cube, self.push_rot, 0.5, 0.5)
        self.stage = 3
      elif self.stage == 3:
        # lift up to push next
        self.target_push = (self.push_pos_lift, self.push_rot, 0.5, 0.5)
        self.stage = 4
      elif self.stage == 4:
        # to pre push start pos
        self.target_push = (self.pre_push_start_pos_tri, self.push_rot_tri, 0.5, 0.5)
        self.stage = 5
      elif self.stage == 5:
        # to push start pos
        self.target_push = (self.push_start_pos_tri, self.push_rot_tri, 0.5, 0.5)
        self.stage = 6
      elif self.stage == 6:
        # to push end pos
        self.target_push = (self.push_end_pos_tri, self.push_rot_tri, 0.5, 0.5)
        self.stage = 0

  def setFlag(self, flag):
    self.arrange_stack = flag
  def getNextAction(self):

    if self.arrange_stack:
      if self.env.current_episode_steps == 1:
        self.pick_place_stage = 0
        self.target_obj = None
        self.current_target = None

      if self.current_target is not None:
        return self.getNextActionToCurrentTarget()
      else:
        self.setNewTarget()
        return self.getNextActionToCurrentTarget()

    else:
      if self.env.current_episode_steps == 1:
        self.target_push = None
        self.stage = 0
      if self.target_push is not None:
        return self.getNextActionToCurrentTarget()
      else:
        self.setNewTarget()
        return self.getNextActionToCurrentTarget()

  def getStepsLeft(self):
    return 100