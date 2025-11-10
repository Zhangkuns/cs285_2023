from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import hw5.cs285.infrastructure.pytorch_util as ptu
from hw5.cs285.agents.dqn_agent import DQNAgent


class CQLAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        cql_alpha: float,
        cql_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )
        self.cql_alpha = cql_alpha
        self.cql_temperature = cql_temperature

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: bool,
    ) -> Tuple[torch.Tensor, dict, dict]:
        loss, metrics, variables = super().compute_critic_loss(
            obs,
            action,
            reward,
            next_obs,
            done,
        )

        # TODO(student): modify the loss to implement CQL
        # Hint: `variables` includes qa_values and q_values from your CQL implementation
        # CQL regularizer: alpha * ( tau*logsumexp(Q/tau) - Q(s, a_data) )
        q_values: torch.Tensor = variables["q_values"]  # (B,1)
        qa_values: torch.Tensor = variables["qa_values"]  # (B, num_actions)

        # softmax over actions with temperature
        lse = torch.logsumexp(qa_values / self.cql_temperature, dim=1)  # (B,)
        softmax_term = self.cql_temperature * lse  # (B,)

        cql_term = (softmax_term - q_values.squeeze(-1)).mean()  # scalar
        cql_loss = self.cql_alpha * cql_term

        loss = loss + cql_loss

        metrics["cql/term"] = cql_term.item()
        metrics["cql/loss"] = cql_loss.item()

        return loss, metrics, variables
