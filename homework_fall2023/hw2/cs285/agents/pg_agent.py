from typing import Optional, Sequence
import numpy as np
import torch

from hw2.cs285.networks.policies import MLPPolicyPG
from hw2.cs285.networks.critics import ValueCritic
from hw2.cs285.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        # q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)
        q_values = self._calculate_q_vals(rewards)

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        obs = np.concatenate(obs)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        terminals = np.concatenate(terminals)
        q_values = np.concatenate(q_values)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage( obs, rewards, q_values, terminals)

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        actor_info = self.actor.update(obs, actions, advantages)
        info: dict = {
            'actor_info': actor_info
        }

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            critic_info: dict = {}
            for i in range(self.baseline_gradient_steps):
                critic_info = self.critic.update(obs, q_values)  # figure it out `info`
                info['critic_info'] = critic_info
            info.update(critic_info)

        return info

    def update_parallel(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        # q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)
        q_values = self._calculate_q_vals(rewards)

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        obs = np.concatenate(obs)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        terminals = np.concatenate(terminals)
        q_values = np.concatenate(q_values)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage_parallel( obs, rewards, q_values, terminals)

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        actor_info = self.actor.update(obs, actions, advantages)
        info: dict = {
            'actor_info': actor_info
        }

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            critic_info: dict = {}
            for i in range(self.baseline_gradient_steps):
                critic_info = self.critic.update(obs, q_values)  # figure it out `info`
                info['critic_info'] = critic_info
            info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        # if not self.use_reward_to_go:
        #     # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
        #     # trajectory at each point.
        #     # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
        #     # TODO: use the helper function self._discounted_return to calculate the Q-values
        #     q_values = ptu.to_numpy(self._discounted_return(rewards))
        # else:
        #     # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
        #     # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        #     # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
        #     q_values = ptu.to_numpy(self._discounted_reward_to_go(rewards))
        all_q_values = []

        for reward_seq in rewards:
            if self.use_reward_to_go:
                q_seq = self._discounted_reward_to_go(reward_seq)
            else:
                q_seq = self._discounted_return(reward_seq)
            all_q_values.append(q_seq)

        # q_values = np.concatenate(all_q_values)
        return all_q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # TODO: if no baseline, then what are the advantages?
            advantages = q_values
        else:
            # TODO: run the critic and use it as a baseline
            values = ptu.to_numpy(self.critic(ptu.from_numpy(obs))).squeeze()
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO: if using a baseline, but not GAE, what are the advantages?
                # 先用 critic 预测 values = V(obs)，形状应与 q_values 相同；
                # values = ptu.to_numpy(self.critic(ptu.from_numpy(obs))).squeeze()
                advantages = q_values - values
            else:
                # TODO: implement GAE
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    if terminals[i]:
                        next_advantage = 0
                    else:
                        next_advantage = advantages[i + 1]
                    delta = rewards[i] +  (1 - terminals[i]) * self.gamma * values[i+1] - values[i]
                    advantages[i] = delta + (1 - terminals[i]) * self.gamma * self.gae_lambda * next_advantage

                # remove dummy advantage
                advantages = advantages[:-1]

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def _estimate_advantage_parallel(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # TODO: if no baseline, then what are the advantages?
            advantages = q_values
        else:
            # TODO: run the critic and use it as a baseline
            values = ptu.to_numpy(self.critic(ptu.from_numpy(obs)))
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO: if using a baseline, but not GAE, what are the advantages?
                # 先用 critic 预测 values = V(obs)，形状应与 q_values 相同；
                # values = ptu.to_numpy(self.critic(ptu.from_numpy(obs))).squeeze()
                advantages = q_values - values
            else:
                # TODO: implement GAE
                batch_size = obs.shape[0]
                T, E = rewards.shape
                # HINT: append a dummy T+1 value for simpler recursive calculation
                v_pad = np.zeros((T + 1, E), dtype=np.float32)
                v_pad[:T] = values
                advantages = np.zeros((T+1, E), dtype=np.float32)

                mask = (1.0 - terminals.astype(np.float32))  # (T,E)

                for t in reversed(range(T)):
                    delta = rewards[t] + self.gamma * v_pad[t + 1] * mask[t] - v_pad[t]
                    advantages[t] = delta + self.gamma * self.gae_lambda * advantages[t + 1] * mask[t]

                advantages = advantages[:T]  # (T,E)

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        discounted_sum = 0.0
        for t, r in enumerate(rewards):
            discounted_sum += (self.gamma ** t) * r
        return np.full_like(rewards, fill_value=discounted_sum, dtype=np.float32)


    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_sum = 0.0
        for t in reversed(range(len(rewards))):
            running_sum = rewards[t] + self.gamma * running_sum
            discounted_rewards[t] = running_sum
        return discounted_rewards