import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


class RRRobotReach(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        
        self.cfg = cfg

        # 任务配置
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.target_radius = self.cfg["env"]["targetRadius"]
        self.max_push_effort = self.cfg["env"]["maxEffort"]

        # 奖励权重
        self.reward_scales = {}
        self.reward_scales["reach"] = self.cfg["env"]["reachReward"]
        self.reward_scales["effort"] = self.cfg["env"]["effortReward"]
        self.reward_scales["velocity"] = self.cfg["env"]["velocityReward"]
        self.reward_scales["success"] = self.cfg["env"]["successReward"]

        # 观测和动作空间
        self.cfg["env"]["numObservations"] = 10  # 2个关节位置 + 2个关节速度 + 3维度的末端位置(其中一个自由度不用) + 3维度的目标位置
        self.cfg["env"]["numActions"] = 2       # 2关节的力矩

        # 在__init__方法中添加可视化控制
        self.enable_target_vis = self.cfg["env"].get("enableTargetVis", False)  # 默认关闭

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        # 初始化目标位置缓冲区
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        # 设置目标位置的范围（根据机械臂的工作空间）
        self.target_pos_range = {
            'x': [ 0.0, 0.0],  # 根据机械臂实际工作空间调整
            'y': [-1.0, 1.0],
            'z': [ 0.05, 1.0]
        }
        
        # 获取gym状态张量
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # 打印刚体状态张量的形状
        print(f"Rigid body state tensor shape: {rigid_body_tensor.shape}")

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 创建张量包装器
        self.root_states = gymtorch.wrap_tensor(actor_root_state_tensor)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # 13表示每个刚体状态的维度：位置(3) + 旋转四元数(4) + 线速度(3) + 角速度(3) = 13
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # 初始化缓冲区
        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_state.clone()

        # 控制张量
        self.efforts = torch.zeros((self.num_envs, self.num_dof), device=self.device)  # 控制两个关节

        # 重置所有环境
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/rr_robot.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01

        rr_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(rr_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(rr_asset)

        # 设置DOF属性
        dof_props = self.gym.get_asset_dof_properties(rr_asset)
        
        # 第一个关节：执行器（力矩控制）
        dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
        dof_props['stiffness'][0] = 0.0
        dof_props['damping'][0] = 0.0
        
        # 第二个关节：被动关节（无驱动）
        dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
        dof_props['stiffness'][1] = 0.0
        dof_props['damping'][1] = 0.1  # 添加少量阻尼

        # 获取link2和eef的索引（用于计算奖励和观测）
        self.eef_index = self.gym.find_asset_rigid_body_index(rr_asset, "eef")

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, 0.1)

        self.rr_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # 创建环境
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            # 创建机器人
            rr_handle = self.gym.create_actor(env_ptr, rr_asset, start_pose, "rr_robot", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, rr_handle, dof_props)

            self.envs.append(env_ptr)
            self.rr_handles.append(rr_handle)

        # 只在启用可视化时创建marker
        if self.enable_target_vis:
            # Step 1: 创建一个小球资产（marker）
            marker_radius = 0.01
            marker_options = gymapi.AssetOptions()
            marker_options.fix_base_link = True
            marker_asset = self.gym.create_sphere(self.sim, marker_radius, marker_options)

            self.marker_handles = []
            for i in range(self.num_envs):
                env_ptr = self.envs[i]
                marker_pose = gymapi.Transform()
                marker_pose.p = gymapi.Vec3(0.0, 0.0, 0.5)

                marker_handle = self.gym.create_actor(
                    env_ptr, marker_asset, marker_pose, "marker", 9999, 0, 0
                )

                self.gym.set_rigid_body_color(
                    env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 1.0, 0.0)
                )

                self.marker_handles.append(marker_handle)

    def update_marker_positions(self):
        """更新marker位置到目标位置（仅在启用可视化时）"""
        if not self.enable_target_vis:
            return
        
        # 刷新根状态张量
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        # 每个环境有2个actor：机器人(索引0)和marker(索引1)
        num_actors_per_env = 2
        
        # 为每个环境更新marker的根状态
        for i in range(self.num_envs):
            # 计算marker在全局根状态张量中的索引
            marker_global_idx = i * num_actors_per_env + 1  # 每个环境的第二个actor是marker
            
            # 更新marker在根状态张量中的位置
            self.root_states[marker_global_idx, 0:3] = self.target_pos[i]  # 位置(x,y,z)
            self.root_states[marker_global_idx, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)  # 四元数(x,y,z,w)
            self.root_states[marker_global_idx, 7:10] = 0.0   # 线速度
            self.root_states[marker_global_idx, 10:13] = 0.0  # 角速度
        
        # 创建所有marker的全局索引
        marker_indices = torch.arange(self.num_envs, device=self.device) * num_actors_per_env + 1
        marker_indices = marker_indices.to(dtype=torch.int32)
        
        # 使用正确的API方法更新所有marker的位置
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(marker_indices),
            len(marker_indices)
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 获取末端执行器位置
        eef_pos = self.rigid_body_states[:, self.eef_index, 0:3]

        # 观测：关节位置和速度 + 末端位置 + 目标位置
        self.obs_buf = torch.cat([
            self.dof_pos,  # 2个关节位置
            self.dof_vel,  # 2个关节速度
            eef_pos,       # 3个末端位置
            self.target_pos  # 3个目标位置
        ], dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        # 随机初始化关节位置
        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        # 随机生成新的目标位置
        new_target_pos = torch.zeros((len(env_ids), 3), device=self.device)
        # print(f"new_target_pos shape: {new_target_pos.shape}")
        new_target_pos[:, 0] = torch_rand_float(
            self.target_pos_range['x'][0], 
            self.target_pos_range['x'][1], 
            (len(env_ids),1), 
            device=self.device
        ).squeeze(-1)
        new_target_pos[:, 1] = torch_rand_float(
            self.target_pos_range['y'][0], 
            self.target_pos_range['y'][1], 
            (len(env_ids),1), 
            device=self.device
        ).squeeze(-1)
        new_target_pos[:, 2] = torch_rand_float(
            self.target_pos_range['z'][0], 
            self.target_pos_range['z'][1], 
            (len(env_ids),1), 
            device=self.device
        ).squeeze(-1)
        # set a same target position for all envs
        # new_target_pos = torch.tensor([0.0, -0.5, 0.5], device=self.device).repeat(len(env_ids), 1)
        
        # 更新目标位置
        self.target_pos[env_ids] = new_target_pos

        self.dof_pos[env_ids] = positions
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
        # 重置距离跟踪
        if hasattr(self, 'prev_dist_to_target'):
            # 计算新的初始距离
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            eef_pos = self.rigid_body_states[:, self.eef_index, 0:3]
            new_dist = torch.norm(eef_pos - self.target_pos, dim=-1)
            self.prev_dist_to_target[env_ids] = new_dist[env_ids]

    def pre_physics_step(self, actions):
        # actions包含两个关节的力矩
        self.efforts[:, 0] = actions[:, 0] * self.max_push_effort
        # 创建第二个关节的力矩
        self.efforts[:, 1] = actions[:, 1] * self.max_push_effort
        
        # 创建完整的力矩向量（第二个关节力矩为0）
        full_efforts = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        full_efforts[:, 0] = self.efforts[:, 0]
        full_efforts[:, 1] = self.efforts[:, 1]
        
        # 应用力矩
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(full_efforts))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()
        self.update_marker_positions()

    # @torch.jit.script
    def compute_reward(self):
      self.gym.refresh_dof_state_tensor(self.sim)
      self.gym.refresh_rigid_body_state_tensor(self.sim)

      # 获取末端执行器位置
      eef_pos = self.rigid_body_states[:, self.eef_index, 0:3]  # 获取末端执行器的3D位置(x,y,z)
      
      # 计算到目标位置的距离
      dist_to_target = torch.norm(eef_pos - self.target_pos, dim=-1)
      
      # 改进的奖励函数
      # 1. 基于距离的奖励（使用指数衰减而不是线性）
      reach_reward = torch.exp(-2.0 * dist_to_target)
      
      # 2. 成功到达奖励
      success_reward = torch.where(dist_to_target < self.target_radius, 
                                 torch.ones_like(dist_to_target), 
                                 torch.zeros_like(dist_to_target))
      
      # 3. 距离改进奖励（鼓励朝目标移动）
      if hasattr(self, 'prev_dist_to_target'):
          dist_improvement = self.prev_dist_to_target - dist_to_target
          improvement_reward = torch.clamp(dist_improvement * 10.0, -1.0, 1.0)
      else:
          improvement_reward = torch.zeros_like(dist_to_target)
      self.prev_dist_to_target = dist_to_target.clone()
      
      # 惩罚过大的力矩
      effort_penalty = torch.sum(torch.square(self.efforts), dim=-1)
      
      # 惩罚过大的速度
      velocity_penalty = torch.sum(torch.square(self.dof_vel), dim=-1)
      
      # 总奖励
      self.rew_buf = (self.reward_scales["reach"] * reach_reward + 
                    self.reward_scales["success"] * success_reward +
                    improvement_reward -
                    self.reward_scales["effort"] * effort_penalty - 
                    self.reward_scales["velocity"] * velocity_penalty)

      # 重置条件
      self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, 
                                torch.ones_like(self.reset_buf), self.reset_buf)
      
      # 如果到达目标位置，提前重置（放宽阈值）
      self.reset_buf = torch.where(dist_to_target < self.target_radius,  # 使用配置的半径
                                torch.ones_like(self.reset_buf), self.reset_buf)