import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


class R3RobotReach(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        
        self.cfg = cfg

        # 任务配置
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.target_radius = self.cfg["env"]["targetRadius"]
        self.max_push_effort = self.cfg["env"]["maxEffort"]
        
        # 机械臂参数
        self.joint_damping = self.cfg["env"]["jointDamping"]
        self.joint_friction = self.cfg["env"]["jointFriction"]
        self.max_joint_vel = self.cfg["env"]["maxJointVel"]

        # 奖励权重
        self.reward_scales = {}
        self.reward_scales["reach"] = self.cfg["env"]["reachReward"]
        self.reward_scales["effort"] = self.cfg["env"]["effortReward"]
        self.reward_scales["velocity"] = self.cfg["env"]["velocityReward"]
        self.reward_scales["success"] = self.cfg["env"]["successReward"]
        self.reward_scales["orientation"] = self.cfg["env"].get("orientationReward", 0.0)

        # 观测和动作空间 - 3DOF机械臂
        self.cfg["env"]["numObservations"] = 12  # 3个关节位置 + 3个关节速度 + 3维末端位置 + 3维目标位置
        self.cfg["env"]["numActions"] = 3       # 3关节的力矩

        # 可视化控制
        self.enable_target_vis = self.cfg["env"].get("enableTargetVis", False)

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        # 初始化目标位置缓冲区
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        # 设置R3机械臂的工作空间范围 (总长度约1.05m)
        self.target_pos_range = {
            'x': [-0.8, 0.8],   # 左右摆动范围
            'y': [-0.8, 0.8],   # 前后伸展范围  
            'z': [0.2, 1.2]     # 高度范围
        }
        
        # 获取gym状态张量
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        print(f"R3机械臂刚体状态张量形状: {rigid_body_tensor.shape}")
        print(f"R3机械臂DOF数量: {self.num_dof}")

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

        # 控制张量 - 3个关节
        self.efforts = torch.zeros((self.num_envs, self.num_dof), device=self.device)

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
        asset_file = "urdf/r3_robot.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = self.joint_damping
        asset_options.linear_damping = self.joint_damping

        r3_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(r3_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(r3_asset)
        
        print(f"R3机械臂加载完成 - DOF: {self.num_dof}, Bodies: {self.num_bodies}")

        # 设置DOF属性 - 3个关节都是力矩控制
        dof_props = self.gym.get_asset_dof_properties(r3_asset)
        
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            dof_props['stiffness'][i] = 0.0
            dof_props['damping'][i] = self.joint_damping
            dof_props['friction'][i] = self.joint_friction
            dof_props['velocity'][i] = self.max_joint_vel

        # 获取末端执行器索引
        self.eef_index = self.gym.find_asset_rigid_body_index(r3_asset, "eef")
        print(f"末端执行器索引: {self.eef_index}")

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, 0.1)

        self.r3_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # 创建环境
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            # 创建R3机械臂
            r3_handle = self.gym.create_actor(env_ptr, r3_asset, start_pose, "r3_robot", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, r3_handle, dof_props)

            self.envs.append(env_ptr)
            self.r3_handles.append(r3_handle)

        # 创建目标可视化标记
        if self.enable_target_vis:
            self._create_target_markers()

    def _create_target_markers(self):
        """创建目标位置的可视化标记"""
        marker_radius = 0.02
        marker_options = gymapi.AssetOptions()
        marker_options.fix_base_link = True
        marker_asset = self.gym.create_sphere(self.sim, marker_radius, marker_options)

        self.marker_handles = []
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            marker_pose = gymapi.Transform()
            marker_pose.p = gymapi.Vec3(0.0, 0.0, 0.5)

            marker_handle = self.gym.create_actor(
                env_ptr, marker_asset, marker_pose, "target_marker", 9999, 0, 0
            )

            # 设置标记颜色为绿色
            self.gym.set_rigid_body_color(
                env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 1.0, 0.0)
            )

            self.marker_handles.append(marker_handle)

    def update_marker_positions(self):
        """更新目标标记位置"""
        if not self.enable_target_vis:
            return
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        num_actors_per_env = 2  # 机械臂 + 标记
        
        for i in range(self.num_envs):
            marker_global_idx = i * num_actors_per_env + 1
            
            self.root_states[marker_global_idx, 0:3] = self.target_pos[i]
            self.root_states[marker_global_idx, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            self.root_states[marker_global_idx, 7:10] = 0.0
            self.root_states[marker_global_idx, 10:13] = 0.0
        
        marker_indices = torch.arange(self.num_envs, device=self.device) * num_actors_per_env + 1
        marker_indices = marker_indices.to(dtype=torch.int32)
        
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

        # 观测：关节状态 + 末端位置 + 目标位置
        self.obs_buf = torch.cat([
            self.dof_pos,       # 3个关节位置
            self.dof_vel,       # 3个关节速度  
            eef_pos,            # 3个末端位置
            self.target_pos     # 3个目标位置
        ], dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        # 随机初始化关节位置 - 为R3机械臂设置合理的初始范围
        positions = torch.zeros((len(env_ids), self.num_dof), device=self.device)
        positions[:, 0] = torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(-1)  # 基座旋转
        positions[:, 1] = torch_rand_float(-0.3, 0.3, (len(env_ids), 1), device=self.device).squeeze(-1)  # 肩部俯仰
        positions[:, 2] = torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(-1)  # 肘部俯仰
        
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        # 生成新的目标位置
        new_target_pos = torch.zeros((len(env_ids), 3), device=self.device)
        new_target_pos[:, 0] = torch_rand_float(
            self.target_pos_range['x'][0], 
            self.target_pos_range['x'][1], 
            (len(env_ids), 1), 
            device=self.device
        ).squeeze(-1)
        new_target_pos[:, 1] = torch_rand_float(
            self.target_pos_range['y'][0], 
            self.target_pos_range['y'][1], 
            (len(env_ids), 1), 
            device=self.device
        ).squeeze(-1)
        new_target_pos[:, 2] = torch_rand_float(
            self.target_pos_range['z'][0], 
            self.target_pos_range['z'][1], 
            (len(env_ids), 1), 
            device=self.device
        ).squeeze(-1)
        new_target_pos[:, 0] = 0.2
        new_target_pos[:, 1] = 0.2
        new_target_pos[:, 2] = 0.9
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
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            eef_pos = self.rigid_body_states[:, self.eef_index, 0:3]
            new_dist = torch.norm(eef_pos - self.target_pos, dim=-1)
            self.prev_dist_to_target[env_ids] = new_dist[env_ids]

    def pre_physics_step(self, actions):
        # actions包含3个关节的力矩
        for i in range(self.num_dof):
            self.efforts[:, i] = actions[:, i] * self.max_push_effort
        
        # 应用力矩
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.efforts))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()
        self.update_marker_positions()

    def compute_reward(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 获取末端执行器位置
        eef_pos = self.rigid_body_states[:, self.eef_index, 0:3]
        
        # 计算到目标位置的距离
        dist_to_target = torch.norm(eef_pos - self.target_pos, dim=-1)
        
        # 1. 基于距离的奖励（指数衰减）
        reach_reward = torch.exp(-3.0 * dist_to_target)
        
        # 2. 成功到达奖励
        success_reward = torch.where(dist_to_target < self.target_radius, 
                                   torch.ones_like(dist_to_target), 
                                   torch.zeros_like(dist_to_target))
        
        # 3. 距离改进奖励
        if hasattr(self, 'prev_dist_to_target'):
            dist_improvement = self.prev_dist_to_target - dist_to_target
            improvement_reward = torch.clamp(dist_improvement * 15.0, -1.0, 1.0)
        else:
            improvement_reward = torch.zeros_like(dist_to_target)
        self.prev_dist_to_target = dist_to_target.clone()
        
        # 4. 力矩惩罚
        effort_penalty = torch.sum(torch.square(self.efforts), dim=-1)
        
        # 5. 速度惩罚
        velocity_penalty = torch.sum(torch.square(self.dof_vel), dim=-1)
        
        # 6. 关节限制惩罚 (保持关节在合理范围内)
        joint_limit_penalty = torch.zeros_like(dist_to_target)
        # 对超出合理范围的关节角度进行惩罚
        joint_limit_penalty += torch.sum(torch.clamp(torch.abs(self.dof_pos) - 2.0, 0.0, float('inf')), dim=-1)
        
        # 计算加权奖励组件
        weighted_reach_reward = self.reward_scales["reach"] * reach_reward
        weighted_success_reward = self.reward_scales["success"] * success_reward
        weighted_effort_penalty = self.reward_scales["effort"] * effort_penalty
        weighted_velocity_penalty = self.reward_scales["velocity"] * velocity_penalty
        weighted_joint_limit_penalty = 0.1 * joint_limit_penalty
        
        # 总奖励
        self.rew_buf = (weighted_reach_reward + 
                      weighted_success_reward +
                      improvement_reward -
                      weighted_effort_penalty - 
                      weighted_velocity_penalty -
                      weighted_joint_limit_penalty)

        # 记录详细奖励信息到TensorBoard
        self.extras["rewards/distance_to_target"] = dist_to_target.mean()
        self.extras["rewards/reach_reward"] = weighted_reach_reward.mean()
        self.extras["rewards/success_reward"] = weighted_success_reward.mean()
        self.extras["rewards/improvement_reward"] = improvement_reward.mean()
        self.extras["losses/effort_penalty"] = weighted_effort_penalty.mean()
        self.extras["losses/velocity_penalty"] = weighted_velocity_penalty.mean()
        self.extras["losses/joint_limit_penalty"] = weighted_joint_limit_penalty.mean()
        self.extras["rewards/total_reward"] = self.rew_buf.mean()
        
        # 记录成功率统计
        success_rate = (dist_to_target < self.target_radius).float().mean()
        self.extras["metrics/success_rate"] = success_rate
        self.extras["metrics/average_distance"] = dist_to_target.mean()
        
        # 记录关节状态
        self.extras["robot/joint_pos_mean"] = torch.abs(self.dof_pos).mean()
        self.extras["robot/joint_vel_mean"] = torch.abs(self.dof_vel).mean()
        self.extras["robot/effort_mean"] = torch.abs(self.efforts).mean()
        
        # 重置条件
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, 
                                   torch.ones_like(self.reset_buf), self.reset_buf)
        
        # 成功到达目标时重置
        self.reset_buf = torch.where(dist_to_target < self.target_radius,
                                   torch.ones_like(self.reset_buf), self.reset_buf)


 