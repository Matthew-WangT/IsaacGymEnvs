import os
import isaacgym
from isaacgym import gymapi
import numpy as np

def main():
    # 初始化gym
    gym = gymapi.acquire_gym()

    # 创建sim
    sim_params = gymapi.SimParams()
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    # 创建地面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    # 创建环境
    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

    # 加载URDF - 修正路径
    asset_root = os.path.dirname(os.path.abspath(__file__))
    asset_file = "rr_robot.urdf"
    
    # 确保路径存在
    full_path = os.path.join(asset_root, asset_file)
    print(f"full_path: {full_path}")
    if not os.path.exists(full_path):
        print(f"错误：找不到URDF文件: {full_path}")
        return

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = False
    asset_options.collapse_fixed_joints = False
    asset_options.disable_gravity = False
    asset_options.thickness = 0.001
    asset_options.angular_damping = 0.01
    asset_options.linear_damping = 0.01

    try:
        robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    except Exception as e:
        print(f"加载URDF文件时出错: {e}")
        return

    # 设置机器人初始位置
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.1)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # 创建机器人actor
    robot_handle = gym.create_actor(env, robot_asset, pose, "robot", 0, 0)

    # 创建查看器
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("错误：无法创建查看器")
        return

    # 设置相机视角
    cam_pos = gymapi.Vec3(1.0, 1.0, 1.0)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # 主循环
    while not gym.query_viewer_has_closed(viewer):
        # 步进模拟
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()
