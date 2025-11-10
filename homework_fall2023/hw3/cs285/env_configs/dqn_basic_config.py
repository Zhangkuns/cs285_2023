from typing import Optional, Tuple

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

import numpy as np
import torch
import torch.nn as nn

from hw3.cs285.env_configs.schedule import (
    LinearSchedule,
    PiecewiseSchedule,
    ConstantSchedule,
)
import hw3.cs285.infrastructure.pytorch_util as ptu

def basic_dqn_config(
    env_name: str,
    exp_name: Optional[str] = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 1e-3,
    total_steps: int = 300000,
    discount: float = 0.99,
    target_update_period: int = 1000,
    clip_grad_norm: Optional[float] = None,
    use_double_q: bool = False,
    learning_starts: int = 20000,
    batch_size: int = 128,
    exploration_type: str = "linear_decay",  # 新增参数
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.02,  # 新增参数
    epsilon_decay_steps: Optional[int] = None,  # 新增参数
    **kwargs
):
    def make_critic(observation_shape: Tuple[int, ...], num_actions: int) -> nn.Module:
        return ptu.build_mlp(
            input_size=np.prod(observation_shape),
            output_size=num_actions,
            n_layers=num_layers,
            size=hidden_size,
        )

    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=learning_rate)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    # exploration_schedule = PiecewiseSchedule(
    #     [
    #         (0, 1),
    #         (total_steps * 0.1, 0.02),
    #     ],
    #     outside_value=0.02,
    # )

    # 根据exploration_type选择不同的策略
    if exploration_type == "linear_decay":
        # 线性衰减（原始版本）
        exploration_schedule = PiecewiseSchedule(
            [
                (0, epsilon_start),
                (total_steps * 0.1, epsilon_end),
            ],
            outside_value=epsilon_end,
        )
        exp_suffix = f"_linear_{epsilon_start}to{epsilon_end}"

    elif exploration_type == "constant":
        # 恒定epsilon
        exploration_schedule = ConstantSchedule(epsilon_start)
        exp_suffix = f"_const_{epsilon_start}"

    elif exploration_type == "step_decay":
        # 阶梯式衰减
        exploration_schedule = PiecewiseSchedule(
            [
                (0, 1.0),
                (total_steps * 0.1, 0.5),
                (total_steps * 0.3, 0.1),
                (total_steps * 0.5, 0.05),
                (total_steps * 0.7, 0.01),
            ],
            outside_value=0.01,
        )
        exp_suffix = "_step_decay"

    elif exploration_type == "slow_decay":
        # 缓慢衰减（更多探索）
        exploration_schedule = PiecewiseSchedule(
            [
                (0, 1.0),
                (total_steps * 0.5, 0.1),  # 50%的时间才降到0.1
                (total_steps * 0.8, 0.02),  # 80%的时间才降到0.02
            ],
            outside_value=0.02,
        )
        exp_suffix = "_slow_decay"

    elif exploration_type == "fast_decay":
        # 快速衰减（更多利用）
        exploration_schedule = PiecewiseSchedule(
            [
                (0, 1.0),
                (total_steps * 0.02, 0.1),  # 2%的时间就降到0.1
                (total_steps * 0.05, 0.02),  # 5%的时间就降到0.02
            ],
            outside_value=0.02,
        )
        exp_suffix = "_fast_decay"

    else:
        raise ValueError(f"Unknown exploration_type: {exploration_type}")

    def make_env(render: bool = False):
        return RecordEpisodeStatistics(gym.make(env_name, render_mode="rgb_array" if render else None))

    log_string = "{}_{}_s{}_l{}_d{}".format(
        exp_name or "dqn",
        env_name,
        hidden_size,
        num_layers,
        discount,
    )

    if use_double_q:
        log_string += "_doubleq"

    return {
        "agent_kwargs": {
            "make_critic": make_critic,
            "make_optimizer": make_optimizer,
            "make_lr_schedule": make_lr_schedule,
            "discount": discount,
            "target_update_period": target_update_period,
            "clip_grad_norm": clip_grad_norm,
            "use_double_q": use_double_q,
        },
        "exploration_schedule": exploration_schedule,
        "log_name": log_string,
        "make_env": make_env,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "learning_starts": learning_starts,
        **kwargs,
    }
