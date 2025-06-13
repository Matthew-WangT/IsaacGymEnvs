import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


class KuavoArmReach(VecTask):

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

        # 观测和动作空间 - Kuavo机械臂有4个DOF
        self.cfg["env"]["numObservations"] = 14  # 4个关节位置 + 4个关节速度 + 3维末端位置 + 3维目标位置
        self.cfg["env"]["numActions"] = 4       # 4个关节的力矩

        # 可视化控制 - 暂时禁用以避免段错误
        self.enable_target_vis = False

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        # 初始化目标位置缓冲区
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        # 设置Kuavo机械臂的工作空间范围
        self.target_pos_range = {
            'x': [-0.8, 0.8],   # 左右摆动范围
            'y': [-0.8, 0.8],   # 前后伸展范围  
            'z': [0.2, 1.2]     # 高度范围
        }
        
        # 获取gym状态张量
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        print(f"Kuavo机械臂刚体状态张量形状: {rigid_body_tensor.shape}")
        print(f"Kuavo机械臂DOF数量: {self.num_dof}")

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

        # 控制张量 - 4个关节
        self.efforts = torch.zeros((self.num_envs, self.num_dof), device=self.device)

        # 初始化距离跟踪
        self.prev_dist_to_target = None

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
        asset_file = "urdf/kuavo/urdf/kuavo_arm_simplified.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = self.joint_damping
        asset_options.linear_damping = self.joint_damping
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        asset_options.use_mesh_materials = False
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.replace_cylinder_with_capsule = True

        print(f"正在加载URDF文件: {os.path.join(asset_root, asset_file)}")
        
        if not os.path.exists(os.path.join(asset_root, asset_file)):
            raise FileNotFoundError(f"URDF文件不存在: {os.path.join(asset_root, asset_file)}")
            
        kuavo_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        if kuavo_asset is None:
            raise RuntimeError("无法加载Kuavo机械臂资产")
            
        self.num_dof = self.gym.get_asset_dof_count(kuavo_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(kuavo_asset)
        
        print(f"Kuavo机械臂加载完成 - DOF: {self.num_dof}, Bodies: {self.num_bodies}")

        # 设置DOF属性 - 所有关节都是力矩控制
        dof_props = self.gym.get_asset_dof_properties(kuavo_asset)
        
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            dof_props['stiffness'][i] = 0.0
            dof_props['damping'][i] = self.joint_damping
            dof_props['friction'][i] = self.joint_friction
            dof_props['velocity'][i] = self.max_joint_vel

        # 获取末端执行器索引
        self.eef_index = self.gym.find_asset_rigid_body_index(kuavo_asset, "eef_sphere")
        print(f"末端执行器索引: {self.eef_index}")
        
        if self.eef_index == -1:
            print("警告: 未找到名为 'eef_sphere' 的刚体，列出所有刚体名称:")
            for i in range(self.num_bodies):
                body_name = self.gym.get_asset_rigid_body_name(kuavo_asset, i)
                print(f"  刚体 {i}: {body_name}")
            raise RuntimeError("无法找到末端执行器 'eef_sphere'")

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, 0.1)

        self.kuavo_handles = []
        self.envs = []
        
        print(f"开始创建 {self.num_envs} 个环境...")
        
        for i in range(self.num_envs):
            try:
                # 创建环境
                env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
                if env_ptr is None:
                    raise RuntimeError(f"无法创建环境 {i}")
                print(f"环境 {i} 创建成功")
                # 创建Kuavo机械臂
                # print(f"准备创建actor {i}...")
                print(f"start_pose: p=({start_pose.p.x}, {start_pose.p.y}, {start_pose.p.z})")
                kuavo_handle = self.gym.create_actor(env_ptr, kuavo_asset, start_pose, "kuavo_robot", i, 0, 0)
                # print(f"actor {i} 创建完成")
                if kuavo_handle is None:
                    raise RuntimeError(f"无法在环境 {i} 中创建机械臂")
                
                # 设置DOF属性
                print(f"设置DOF属性")
                print(f"DOF属性: {dof_props}")
                self.gym.set_actor_dof_properties(env_ptr, kuavo_handle, dof_props)

                self.envs.append(env_ptr)
                self.kuavo_handles.append(kuavo_handle)
                
                if (i + 1) % 100 == 0:
                    print(f"已创建 {i + 1} 个环境...")
                    
            except Exception as e:
                print(f"创建环境 {i} 时出错: {e}")
                raise
                
        print(f"Kuavo机械臂创建完成 - 环境数量: {self.num_envs}")

    def _create_target_markers(self):
        """创建目标位置的可视化标记 - 已禁用"""
        pass

    def update_marker_positions(self):
        """更新目标标记位置 - 已禁用"""
        pass

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 获取末端执行器位置
        eef_pos = self.rigid_body_states[:, self.eef_index, 0:3]

        # 观测：关节状态 + 末端位置 + 目标位置
        self.obs_buf = torch.cat([
            self.dof_pos,       # 4个关节位置
            self.dof_vel,       # 4个关节速度  
            eef_pos,            # 3个末端位置
            self.target_pos     # 3个目标位置
        ], dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        # 随机初始化关节位置 - 为Kuavo机械臂设置合理的初始范围
        positions = torch.zeros((len(env_ids), self.num_dof), device=self.device)
        # 根据URDF中的关节限制设置合理的初始角度范围
        positions[:, 0] = torch_rand_float(-1.0, 0.5, (len(env_ids), 1), device=self.device).squeeze(-1)  # zarm_l1_joint
        positions[:, 1] = torch_rand_float(-0.2, 1.0, (len(env_ids), 1), device=self.device).squeeze(-1)  # zarm_l2_joint
        positions[:, 2] = torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(-1)  # zarm_l3_joint
        positions[:, 3] = torch_rand_float(-1.5, -0.2, (len(env_ids), 1), device=self.device).squeeze(-1) # zarm_l4_joint
        
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
        if self.prev_dist_to_target is None:
            self.prev_dist_to_target = torch.ones((self.num_envs,), device=self.device)
            
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        eef_pos = self.rigid_body_states[:, self.eef_index, 0:3]
        new_dist = torch.norm(eef_pos - self.target_pos, dim=-1)
        self.prev_dist_to_target[env_ids] = new_dist[env_ids]

    def pre_physics_step(self, actions):
        # actions包含4个关节的力矩
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
        dist_improvement = self.prev_dist_to_target - dist_to_target
        improvement_reward = torch.clamp(dist_improvement * 15.0, -1.0, 1.0)
        self.prev_dist_to_target = dist_to_target.clone()
        
        # 4. 力矩惩罚
        effort_penalty = torch.sum(torch.square(self.efforts), dim=-1)
        
        # 5. 速度惩罚
        velocity_penalty = torch.sum(torch.square(self.dof_vel), dim=-1)
        
        # 6. 关节限制惩罚 (保持关节在合理范围内)
        joint_limit_penalty = torch.zeros_like(dist_to_target)
        
        # 根据URDF中的关节限制设置惩罚
        joint_limits = torch.tensor([
            [-3.14159265358979, 1.5707963267949],    # zarm_l1_joint
            [-0.349065850398866, 2.0943951023932],   # zarm_l2_joint
            [-1.5707963267949, 1.5707963267949],     # zarm_l3_joint
            [-2.61799387799149, 0.0]                 # zarm_l4_joint
        ], device=self.device)
        
        for i in range(self.num_dof):
            below_min = torch.clamp(joint_limits[i, 0] - self.dof_pos[:, i], 0.0, float('inf'))
            above_max = torch.clamp(self.dof_pos[:, i] - joint_limits[i, 1], 0.0, float('inf'))
            joint_limit_penalty += below_min + above_max
        
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