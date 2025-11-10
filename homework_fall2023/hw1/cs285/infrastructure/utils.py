"""A
Some miscellaneous utility functions

Functions to edit:
    1. sample_trajectory
"""

from collections import OrderedDict
import cv2
import numpy as np
import time

from hw1.cs285.infrastructure import pytorch_util as ptu


def sample_trajectory(env, policy, max_path_length, render=False):
    """Sample a rollout in the environment from a policy."""
    # 智能体与环境交互采样一条轨迹（episode）

    # initialize env for the beginning of a new rollout
    ob =  env.reset() # TODO: initial observation after resetting the env

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:

        # render image of the simulated env
        if render:
            if hasattr(env, 'sim'):
                img = env.sim.render(camera_name='track', height=500, width=500)[::-1]
            else:
                img = env.render(mode='single_rgb_array')
            image_obs.append(cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))

        # TODO use the most recent ob to decide what to do
        ac = policy.get_action(ob) # HINT: this is a numpy array

        #####################################
        # ---- 关键规范化处理：转 numpy + 去批量维 + reshape 到正确形状 + float32 ----
        import numpy as np
        try:
            ac = ac.detach().cpu().numpy()
        except AttributeError:
            ac = np.asarray(ac)

        ac = ac.astype(np.float32)

        # 去掉可能的批量维
        if ac.ndim > 1:
            ac = np.squeeze(ac)

        # 保险地 reshape 到环境动作空间形状（例如 (8,)）
        ac = ac.reshape(env.action_space.shape)
        #####################################

        # TODO: take that action and get reward and next ob
        # 调用 Gym 环境的标准接口：
        # next_ob: 下一状态；
        # rew: 当前步奖励；
        # done: 是否结束；
        # _: 其他信息（一般不用）。
        next_ob, rew, done, _ = env.step(ac)

        # TODO rollout can end due to done, or due to max_path_length
        steps += 1
        rollout_done = done or (steps >= max_path_length) # HINT: this is either 0 or 1

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False):
    """Collect rollouts until we have collected min_timesteps_per_batch steps."""
    # 重复调用 sample_trajectory()，直到收集的时间步数 ≥ min_timesteps_per_batch。

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:

        #collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)

        #count steps
        timesteps_this_batch += get_pathlength(path)

    return paths, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False):
    """Collect ntraj rollouts."""
    # 重复调用 sample_trajectory()，直到固定轨迹数量（ntraj）：

    paths = []
    for i in range(ntraj):
        # collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
    return paths


########################################
########################################


def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    # 把若干条轨迹拼接成单个 numpy 数组：
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals


########################################
########################################


def compute_metrics(paths, eval_paths):
    """Compute metrics for logging."""
    # 计算训练和评估的统计指标：
    # 输出包括：
    # 平均回报（AverageReturn）
    # 标准差（StdReturn）
    # 最大 / 最小回报
    # 平均 episode 长度
    # 用于日志记录（logger）。

    # returns, for logging
    train_returns = [path["reward"].sum() for path in paths]
    eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

    # episode lengths, for logging
    train_ep_lens = [len(path["reward"]) for path in paths]
    eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

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


############################################
############################################


def get_pathlength(path):
    return len(path["reward"])
