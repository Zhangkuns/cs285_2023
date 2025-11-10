from collections import OrderedDict
import numpy as np
import copy
from hw2.cs285.networks.policies import MLPPolicy
import gym
import cv2
from hw2.cs285.infrastructure import pytorch_util as ptu
from typing import Dict, Tuple, List

############################################
############################################


def sample_trajectory(
    env: gym.Env, policy: MLPPolicy, max_length: int, render: bool = False
) -> Dict[str, np.ndarray]:
    """Sample a rollout in the environment from a policy."""
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # render an image
        if render:
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
            else:
                img = env.render(mode="single_rgb_array")
            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
            )

        # TODO use the most recent ob and the policy to decide what to do
        ac: np.ndarray = policy.get_action(ob)

        # TODO: use that action to take a step in the environment
        next_ob, rew, done, _ = env.step(ac)

        # TODO rollout can end due to done, or due to max_length
        steps += 1
        rollout_done: bool = done or (steps >= max_length)

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }

def sample_trajectories(
    env: gym.Env,
    policy: MLPPolicy,
    min_timesteps_per_batch: int,
    max_length: int,
    render: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """Collect rollouts using policy until we have collected min_timesteps_per_batch steps."""
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)

        # count steps
        timesteps_this_batch += get_traj_length(traj)
    return trajs, timesteps_this_batch

def sample_n_trajectories(
    env: gym.Env, policy: MLPPolicy, ntraj: int, max_length: int, render: bool = False
):
    """Collect ntraj rollouts."""
    trajs = []
    for _ in range(ntraj):
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)
    return trajs

###############################################
# Parallel env version of the above functions #
###############################################

def _empty_traj_buf():
    return {
        "observation": [],
        "action": [],
        "reward": [],
        "next_observation": [],
        "terminal": [],
        "image_obs": [],  # 保留字段，和单环境版本一致（这里不做并行渲染）
    }

def _pack_traj(buf: Dict[str, list]) -> Dict[str, np.ndarray]:
    return {
        "observation":      np.array(buf["observation"],      dtype=np.float32),
        "image_obs":        np.array(buf["image_obs"],        dtype=np.uint8),
        "reward":           np.array(buf["reward"],           dtype=np.float32),
        "action":           np.array(buf["action"],           dtype=np.float32),
        "next_observation": np.array(buf["next_observation"], dtype=np.float32),
        "terminal":         np.array(buf["terminal"],         dtype=np.float32),
    }

def sample_trajectories_parallel(
    env: gym.vector.VectorEnv,
    policy: MLPPolicy,
    min_timesteps_per_batch: int,
    max_length: int,
    render: bool = False,   # 并行下这里不做渲染，如需视频请用单环境专门采
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """在并行环境下采样，直到累计步数 >= min_timesteps_per_batch。
    返回：List[traj_dict]（逐条轨迹），以及累计 env steps（包含所有并行环境的总步数）。
    """
    assert hasattr(env, "num_envs"), "sample_trajectories_parallel 需要 VectorEnv/AsyncVectorEnv"

    E = env.num_envs
    obs = env.reset()                   # (E, obs_dim) ；Gym 0.25 返回 obs；Gymnasium 返回 (obs, info)

    # 给每个 env 一套独立的缓存，用来“封轨迹”
    buffers = [_empty_traj_buf() for _ in range(E)]
    ep_steps = np.zeros(E, dtype=np.int32)

    trajs: List[Dict[str, np.ndarray]] = []
    timesteps_this_batch = 0

    while timesteps_this_batch < min_timesteps_per_batch:
        actions = policy.get_action(obs)                  # (E, act_dim) 或 (E,)
        step_out = env.step(actions)
        # VectorEnv / AsyncVectorEnv: (next_obs, rewards, dones, infos)
        next_obs, rewards, dones, infos = step_out

        # 将这一时间步写入每个 env 的缓存
        for e in range(E):
            buffers[e]["observation"].append(obs[e])
            buffers[e]["action"].append(actions[e])
            buffers[e]["reward"].append(rewards[e])
            buffers[e]["next_observation"].append(next_obs[e])

            # 本步是否 episode 结束（或达上限）
            ep_steps[e] += 1
            rollout_done = bool(dones[e]) or (ep_steps[e] >= max_length)
            buffers[e]["terminal"].append(rollout_done)

            timesteps_this_batch += 1

            if rollout_done:
                # 把该 env 当前缓存封成一条“单环境轨迹”
                trajs.append(_pack_traj(buffers[e]))
                # 清空该 env 的缓存，重新开始下一条轨迹
                buffers[e] = _empty_traj_buf()
                ep_steps[e] = 0

        obs = next_obs

    return trajs, timesteps_this_batch

def sample_n_trajectories_parallel(
    env: gym.vector.VectorEnv,
    policy: MLPPolicy,
    ntraj: int,
    max_length: int,
    render: bool = False,
) -> List[Dict[str, np.ndarray]]:
    """采样恰好 n 条轨迹（并行收集），用于评估或拍视频（不含并行渲染）。"""
    E = env.num_envs
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs

    buffers = [_empty_traj_buf() for _ in range(E)]
    ep_steps = np.zeros(E, dtype=np.int32)

    trajs: List[Dict[str, np.ndarray]] = []

    while len(trajs) < ntraj:
        actions = policy.get_action(obs)
        next_obs, rewards, dones, infos = env.step(actions)

        for e in range(E):
            buffers[e]["observation"].append(obs[e])
            buffers[e]["action"].append(actions[e])
            buffers[e]["reward"].append(rewards[e])
            buffers[e]["next_observation"].append(next_obs[e])

            ep_steps[e] += 1
            rollout_done = bool(dones[e]) or (ep_steps[e] >= max_length)
            buffers[e]["terminal"].append(rollout_done)

            if rollout_done:
                trajs.append(_pack_traj(buffers[e]))
                if len(trajs) >= ntraj:
                    break
                buffers[e] = _empty_traj_buf()
                ep_steps[e] = 0

        obs = next_obs

    return trajs

def compute_metrics(trajs, eval_trajs):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [traj["reward"].sum() for traj in trajs]
    eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]

    # episode lengths, for logging
    train_ep_lens = [len(traj["reward"]) for traj in trajs]
    eval_ep_lens = [len(eval_traj["reward"]) for eval_traj in eval_trajs]

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


def convert_listofrollouts(trajs):
    """
    Take a list of rollout dictionaries and return separate arrays, where each array is a concatenation of that array
    from across the rollouts.
    """
    observations = np.concatenate([traj["observation"] for traj in trajs])
    actions = np.concatenate([traj["action"] for traj in trajs])
    next_observations = np.concatenate([traj["next_observation"] for traj in trajs])
    terminals = np.concatenate([traj["terminal"] for traj in trajs])
    concatenated_rewards = np.concatenate([traj["reward"] for traj in trajs])
    unconcatenated_rewards = [traj["reward"] for traj in trajs]
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )


def get_traj_length(traj):
    return len(traj["reward"])
