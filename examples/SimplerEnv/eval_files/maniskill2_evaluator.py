"""
Evaluate a model on ManiSkill2 environment.
Fix 4.0: 
1. Coordinate System: Convert World Frame -> Robot Base Frame.
2. Normalization: Apply ONLY to Position (x,y,z). Pass Rotation (r,p,y) as RAW RADIANS.
3. Statistics: Robust loading for Position normalization.
"""

import os
import json
import numpy as np
from transforms3d.euler import quat2euler, quat2mat
from transforms3d.quaternions import qinverse
from pathlib import Path

from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video

# ================= Utils =================

def load_dataset_statistics(ckpt_path):
    ckpt_path = Path(ckpt_path)
    candidates = [
        ckpt_path / "dataset_statistics.json",
        ckpt_path.parent / "dataset_statistics.json",
        ckpt_path.parent.parent / "dataset_statistics.json"
    ]
    
    for path in candidates:
        if path.exists():
            print(f"✅ Found dataset statistics at: {path}")
            with open(path, "r") as f:
                return json.load(f)
    
    print("❌ WARNING: dataset_statistics.json NOT FOUND!")
    return None

def normalize_state(robot_state, stats, unnorm_key="oxe_bridge", debug=False):
    """
    Normalize state based on training distribution analysis:
    - Position (0,1,2): Normalized using q99 stats.
    - Rotation (3,4,5): RAW RADIANS (Identity).
    - Gripper (6): Binary.
    """
    # Default Bounds for WidowX (Fallback for Position)
    # x: 0.1~0.5, y: -0.25~0.25, z: 0.02~0.35
    pos_bounds = np.array([
        [0.1, 0.5],    # x
        [-0.25, 0.25], # y
        [0.02, 0.35]   # z
    ])

    # Try to load real stats for position
    if stats is not None:
        if unnorm_key not in stats:
            unnorm_key = list(stats.keys())[0]
        
        tag_stats = stats[unnorm_key]["state"]
        
        # Check for list-based stats (common in LeRobot)
        if isinstance(tag_stats.get("q01"), list):
            q01_list = np.array(tag_stats["q01"])
            q99_list = np.array(tag_stats["q99"])
            # Indices 0, 1, 2 correspond to x, y, z
            if len(q01_list) >= 3:
                pos_bounds[0, 0], pos_bounds[0, 1] = q01_list[0], q99_list[0]
                pos_bounds[1, 0], pos_bounds[1, 1] = q01_list[1], q99_list[1]
                pos_bounds[2, 0], pos_bounds[2, 1] = q01_list[2], q99_list[2]
                if debug: print(f"   [Debug] Loaded POS stats from list: {pos_bounds}")

    normalized = np.zeros_like(robot_state)
    
    for i in range(7):
        val = robot_state[i]
        
        # --- 1. Position (x, y, z) -> NORMALIZE ---
        if i < 3:
            q01 = pos_bounds[i, 0]
            q99 = pos_bounds[i, 1]
            denom = q99 - q01
            if denom < 1e-8:
                norm_val = val
            else:
                norm_val = 2.0 * (val - q01) / denom - 1.0
            normalized[i] = np.clip(norm_val, -1.0, 1.0)
            
            if debug:
                print(f"   Dim {i} (Pos): Raw {val:.4f} -> Norm {normalized[i]:.4f} (Bounds: {q01:.3f}, {q99:.3f})")
        
        # --- 2. Rotation (r, p, y) -> PASS THROUGH ---
        elif i < 6:
            # Training logs show values like 1.57 and -3.12, implying raw radians.
            # Do NOT normalize.
            normalized[i] = val
            if debug:
                print(f"   Dim {i} (Rot): Raw {val:.4f} -> Keep {normalized[i]:.4f}")

        # --- 3. Gripper -> BINARY ---
        else:
            norm_val = 1.0 if val > 0.04 else 0.0
            normalized[i] = norm_val
            if debug:
                print(f"   Dim {i} (Grp): Raw {val:.4f} -> Bin {normalized[i]:.4f}")

    return normalized

def get_robot_state_in_base_frame(obs):
    """
    Extracts 7D robot state converted to ROBOT BASE FRAME.
    ManiSkill2 provides TCP in World Frame. We must transform it.
    """
    # 1. Get World Frame Data
    # ManiSkill poses are [x, y, z, qx, qy, qz, qw] (Scalar Last)
    tcp_pose_world = obs['extra']['tcp_pose']
    tcp_pos_world = tcp_pose_world[:3]
    tcp_quat_world = tcp_pose_world[3:] # [x, y, z, w]
    
    # 2. Get Base Frame Data
    base_pose = obs['agent']['base_pose']
    base_pos = base_pose[:3]
    base_quat = base_pose[3:] # [x, y, z, w]
    
    # 3. Convert Quaternion to Rotation Matrix
    # transforms3d uses [w, x, y, z] (Scalar First)
    def to_wxyz(q): return np.array([q[3], q[0], q[1], q[2]])
    
    rot_world_to_base = quat2mat(qinverse(to_wxyz(base_quat)))
    
    # 4. Compute Relative Position
    # pos_base = R^T * (pos_world - base_pos)
    rel_pos = rot_world_to_base @ (tcp_pos_world - base_pos)
    
    # 5. Compute Relative Rotation (Euler)
    # R_rel = R_base^T * R_tcp
    rot_tcp = quat2mat(to_wxyz(tcp_quat_world))
    rot_rel = rot_world_to_base @ rot_tcp
    
    # Extract Euler Angles (sxyz matches Bridge convention usually)
    import transforms3d
    r, p, y = transforms3d.euler.mat2euler(rot_rel, axes='sxyz')
    
    # 6. Gripper
    qpos = obs['agent']['qpos']
    gripper_width = np.sum(qpos[-2:]) 
    
    return np.array([rel_pos[0], rel_pos[1], rel_pos[2], r, p, y, gripper_width], dtype=np.float32)


def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
):

    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    dataset_stats = load_dataset_statistics(ckpt_path)
    stats_key = getattr(model, "unnorm_key", "oxe_bridge") 

    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, _ = env.reset(options=env_reset_options)
    is_final_subtask = env.is_final_subtask() 

    if instruction is not None:
        task_description = instruction
    else:
        task_description = env.get_language_instruction()
    print(f"Instruction: {task_description}")

    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    model.reset(task_description)

    timestep = 0
    success = "failure"

    while not (predicted_terminated or truncated):
        # 1. Get State in Base Frame (Critical for Bridge/WidowX)
        robot_state_base = get_robot_state_in_base_frame(obs)
        
        # 2. Normalize (Pos only)
        is_debug = (timestep == 0)
        if is_debug:
             print("\n--- State Processing Debug (Final) ---")
             
        normalized_state = normalize_state(robot_state_base, dataset_stats, unnorm_key=stats_key, debug=is_debug)
        
        if is_debug:
             print(f"State Passed to Model: {normalized_state}")
             print("--------------------------------------\n")

        raw_action, action = model.step(image, task_description, robot_state=normalized_state)
        
        predicted_actions.append(raw_action)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated:
            if not is_final_subtask:
                predicted_terminated = False
                env.advance_to_next_subtask()

        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
        )
        
        success = "success" if done else "failure"
        new_task_description = env.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
        is_final_subtask = env.is_final_subtask()

        if timestep % 10 == 0:
            print(f"Step {timestep}: {info}")

        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images.append(image)
        timestep += 1

    episode_stats = info.get("episode_stats", {})

    # Save video logic
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)

    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return success == "success"


def maniskill2_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []

    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        success_arr.append(run_maniskill2_eval_single_episode(obj_episode_id=obj_episode_id, **kwargs))
                else:
                    raise NotImplementedError()

    return success_arr
