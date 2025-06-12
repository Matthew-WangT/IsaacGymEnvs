import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


class RRRobot(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        
        self.cfg = cfg

        # 任务配置
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_dist = self.cfg["env"]["resetDist"]
        self.max_push_effort = self.cfg["env"]["maxEffort"]

        # 奖励权重
        self.reward_scales = {}
        self.reward_scales["upright"] = self.cfg["env"]["uprightReward"]
        self.reward_scales["effort"] = self.cfg["env"]["effortReward"]
        self.reward_scales["velocity"] = self.cfg["env"]["velocityReward"]

        # 观测和动作空间
        self.cfg["env"]["numObservations"] = 7  # 2个关节位置 + 2个关节速度 + 3个link2方向向量
        self.cfg["env"]["numActions"] = 1       # 只有第一个关节的力矩

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # 获取gym状态张量
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 创建张量包装器
        self.root_states = gymtorch.wrap_tensor(actor_root_state_tensor)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # 初始化缓冲区
        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_state.clone()

        # 控制张量
        self.efforts = torch.zeros_like(self.dof_pos[:, :1])  # 只控制第一个关节

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

        # 获取link2的索引（用于计算奖励）
        self.link2_index = self.gym.find_asset_rigid_body_index(rr_asset, "link2")

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

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 获取link2的方向向量
        link2_quat = self.rigid_body_states[:, self.link2_index, 3:7]
        link2_up = quat_apply(link2_quat, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1))

        # 观测：关节位置和速度 + link2的方向信息
        self.obs_buf = torch.cat([
            self.dof_pos,  # 2个关节位置
            self.dof_vel,  # 2个关节速度
            link2_up       # 3个link2的向上方向分量
        ], dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        # 随机初始化关节位置
        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = positions
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        # actions只包含第一个关节的力矩
        self.efforts[:, 0] = actions.flatten() * self.max_push_effort
        
        # 创建完整的力矩向量（第二个关节力矩为0）
        full_efforts = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        full_efforts[:, 0] = self.efforts[:, 0]
        
        # 应用力矩
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(full_efforts))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

    def compute_reward(self):
        # 获取link2的状态
        link2_states = self.rigid_body_states[:, self.link2_index, :]
        link2_quat = link2_states[:, 3:7]
        
        # 计算link2的方向向量（局部z轴）
        link2_up = quat_apply(link2_quat, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1))
        
        # 目标方向是世界坐标系的z轴（向上）
        target_up = to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1)
        
        # 计算方向相似度（点积）
        upright_reward = torch.sum(link2_up * target_up, dim=-1)
        
        # 惩罚过大的力矩
        effort_penalty = torch.sum(torch.square(self.efforts), dim=-1)
        
        # 惩罚过大的速度
        velocity_penalty = torch.sum(torch.square(self.dof_vel), dim=-1)
        
        # 总奖励
        self.rew_buf = (self.reward_scales["upright"] * upright_reward - 
                       self.reward_scales["effort"] * effort_penalty - 
                       self.reward_scales["velocity"] * velocity_penalty)

        # 重置条件
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, 
                                   torch.ones_like(self.reset_buf), self.reset_buf)
        
        # 如果link2倾斜太多，提前重置
        self.reset_buf = torch.where(upright_reward < 0.3, 
                                   torch.ones_like(self.reset_buf), self.reset_buf)