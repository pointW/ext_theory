import numpy as np
import scipy
from bulletarm.planners.close_loop_planner import CloseLoopPlanner

class CloseLoopBlockColorSortPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.current_target = None
    self.stage = 0
    self.pre_push_start_pos_cube = self.env.workspace.mean(1)
    self.push_start_pos_cube = self.env.workspace.mean(1)
    self.push_end_pos_cube = self.env.workspace.mean(1)
    self.push_rot = 0
    self.pre_push_start_pos_tri = self.env.workspace.mean(1)
    self.push_start_pos_tri = self.env.workspace.mean(1)
    self.push_end_pos_tri = self.env.workspace.mean(1)
    self.push_rot_tri = 0

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    p = self.current_target[2]
    if np.all(np.abs([x, y, z]) < self.dpos) and (not self.random_orientation or np.abs(r) < self.drot):
      self.current_target = None
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
    if self.stage == 0:
      self.setWaypoints()
      # to pre push start pos
      self.current_target = (self.pre_push_start_pos_cube, self.push_rot, 0.5, 0.5)
      self.stage = 1
    elif self.stage == 1:
      # to push start pos
      self.current_target = (self.push_start_pos_cube, self.push_rot, 0.5, 0.5)
      self.stage = 2
    elif self.stage == 2:
      # to push end pos
      self.current_target = (self.push_end_pos_cube, self.push_rot, 0.5, 0.5)
      self.stage = 3

    elif self.stage == 3:
      # lift up to push next
      self.current_target = (self.push_pos_lift, self.push_rot, 0.5, 0.5)
      self.stage = 4

    elif self.stage == 4:
      # to pre push start pos
      self.current_target = (self.pre_push_start_pos_tri, self.push_rot_tri, 0.5, 0.5)
      self.stage = 5
    elif self.stage == 5:
      # to push start pos
      self.current_target = (self.push_start_pos_tri, self.push_rot_tri, 0.5, 0.5)
      self.stage = 6
    elif self.stage == 6:
      # to push end pos
      self.current_target = (self.push_end_pos_tri, self.push_rot_tri, 0.5, 0.5)
      self.stage = 0

  def getNextAction(self):
    if self.env.current_episode_steps == 1:
      self.current_target = None
      self.stage = 0
    if self.current_target is not None:
      return self.getNextActionToCurrentTarget()
    else:
      self.setNewTarget()
      return self.getNextActionToCurrentTarget()

  def getStepsLeft(self):
    return 100