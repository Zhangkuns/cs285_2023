from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn, distributions

import numpy as np

import hw5.cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int], num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        values = self.critic(observation)
        max_action = torch.argmax(values, -1)  # index of action with max value
        n, m = values.shape  # n: batch size, m: #actions
        probs = torch.ones(n, m) * epsilon / m
        probs[torch.arange(n), max_action] = 1 - epsilon + epsilon / m
        dist = distributions.Categorical(probs=probs)
        action = dist.sample()

        return ptu.to_numpy(action).squeeze(0).item()

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict, dict]:
        """
        Compute the loss for the DQN critic.

        Returns:
         - loss: torch.Tensor, the MSE loss for the critic
         - metrics: dict, a dictionary of metrics to log
         - variables: dict, a dictionary of variables that can be used in subsequent calculations
        """

        # TODO(student): paste in your code from HW3, and make sure the return values exist
        # (batch_size,) = reward.shape
        with torch.no_grad():
            next_qa_values = self.target_critic(next_obs)

            if self.use_double_q:
                next_action = torch.argmax(self.critic(next_obs), dim=1).unsqueeze(dim=1)
            else:
                next_action = torch.argmax(next_qa_values, dim=1).unsqueeze(dim=1)

            next_q_values = next_qa_values.gather(dim=1, index=next_action)
            # assert next_q_values.shape == (batch_size,), next_q_values.shape

            target_values = reward.unsqueeze(dim=1) + torch.logical_not(done.unsqueeze(dim=1)) * self.discount * next_q_values
            # assert target_values.shape == (batch_size,), target_values.shape

        qa_values = self.critic(obs)
        q_values = qa_values.gather(dim=1, index=action.unsqueeze(dim=1))
        loss = self.critic_loss(q_values, target_values)

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        loss, metrics, _ = self.compute_critic_loss(obs, action, reward, next_obs, done)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        metrics["grad_norm"] = grad_norm.item()
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return metrics

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): paste in your code from HW3
        # Step 1: 更新 Q 网络（Critic）
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)

        # Step 2: 按周期同步目标网络（Target Network）
        if step % self.target_update_period == 0:
            self.update_target_critic()

        # Step 3: 返回损失和统计信息
        return critic_stats
