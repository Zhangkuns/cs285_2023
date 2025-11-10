from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn, distributions

import numpy as np

import hw3.cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
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

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        values = self.critic(observation)
        max_action = torch.argmax(values, -1)  # index of action with max value
        n, m = values.shape  # n: batch size, m: #actions
        # epsilon-greedy strategy
        # 在权衡开发与探索二者之间， ** ε - 贪心 ** 是一种常用的策略。它表示在智能体做决策时，有一个很小的正数ε( < 1)的概率随机选择一个未知的动作，剩下的1 - ε的概率选择已有动作中价值最大的动作。
        #
        # 假设当前智能体所处的状态为 $s_t \ in S$，其可以选择的动作集合为 $A$。在多臂赌博机中，每一个拉杆可以代表一个动作，智能体在执行某个动作 $a_t \ in A$ 后，将会达到下一个状态 $s_
        # {t + 1}$，并获得对应的收益 $r_t$。
        #
        # 在决策过程中，有ε的概率选择非贪心的动作，即每个动作被选择的概率为 $\frac
        # {\epsilon}{ | A |}$，其中 $ | A |$ 表示动作的数量；也就是说，每个动作都有相同的 $\frac
        # {\epsilon}{ | A |}$ 概率被非贪心地选择。
        #
        # 另外，还有 $1 - \epsilon$ 的概率选择一个贪心策略，因此这个贪心策略被选择的概率为 $1 - \epsilon + \frac
        # {\epsilon}{ | A |}$。
        #
        # 可能有人会问为什么是两项的和，其实很简单。在所有的动作集合 $A$ 中，总会有一个动作是智能体认为的最优动作，即 $a ^ * = \arg\max(
        #     Q(a, s))$。因此，这个动作本身有 $\frac
        # {\epsilon}{ | A |}$ 的概率在探索阶段被选择，还有 $1 - \epsilon$ 的概率在开发阶段被选择。

        # 所以是在生成的动作空间中选择其他的，而不是随机生成其他的动作序列
        # if random.random() < epsilon:
        #     action = torch.randint(0, self.num_actions, (1,)) # 随机生成的动作序列
        # else:
        #     critic_values = self.critic(observation)
        #     action = torch.argmax(critic_values, dim=1)
        # 上述这个为错误的选择
        probs = torch.ones(n, m) * epsilon / m
        probs[torch.arange(n), max_action] = 1 - epsilon + epsilon/m
        dist = distributions.Categorical(probs=probs)
        action = dist.sample()

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(student): compute target values
            # if self.use_double_q:  # to alleviate over-estimating Q-value
            #     values = self.critic(next_obs)
            #     best_action = torch.argmax(values, -1, keepdim=True)
            #     t_values = self.target_critic(next_obs)
            #     next_q_values = torch.gather(t_values, -1, best_action).squeeze()
            # else:
            #     next_q_values = torch.max(self.target_critic(next_obs), -1)[0]  # use target to stabilize training
            #
            # target_values = reward + (~done) * self.discount * next_q_values
            next_qa_values = self.target_critic(next_obs)

            if self.use_double_q:
                next_action = torch.argmax(self.critic(next_obs), dim=1).unsqueeze(dim=1)
            else:
                next_action = torch.argmax(next_qa_values, dim=1).unsqueeze(dim=1)

            next_q_values = next_qa_values.gather(dim=1, index=next_action)
            target_values = reward.unsqueeze(dim=1) + torch.logical_not(done.unsqueeze(dim=1)) * self.discount * next_q_values

        # TODO(student): train the critic with the target values
        qa_values = self.critic(obs)
        q_values = qa_values.gather(dim=1, index=action.unsqueeze(dim=1))
        loss = self.critic_loss(q_values, target_values)


        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

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
        # TODO(student): update the critic, and the target if needed
        # Step 1: 更新 Q 网络（Critic）
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)

        # Step 2: 按周期同步目标网络（Target Network）
        if step % self.target_update_period == 0:
            self.update_target_critic()

        # Step 3: 返回损失和统计信息
        return critic_stats
