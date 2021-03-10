import warnings

import numpy as np
import torch

import pfrl
from pfrl.utils import clip_l2_grad_norm_
from pfrl.utils.batch_states import batch_states
from pfrl.utils.mode_of_distribution import mode_of_distribution


from scipy import signal
def discounted_cumulative_rewards(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class DeepRMSAv2FLX_A2C(pfrl.agents.a2c.A2C):
    """Customized A2C for DeepRMSA_FLX. 

    paper: DeepRMSA: A Deep Reinforcement Learning Framework for Routing, 
            Modulation and Spectrum Assignment in Elastic Optical Networks
            (https://ieeexplore.ieee.org/document/8738827)

    A2C is a synchronous, deterministic variant of Asynchronous Advantage
        Actor Critic (A3C).

    See https://arxiv.org/abs/1708.05144

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optim.Optimizer): optimizer used to train the model
        gamma (float): Discount factor [0,1]
        num_processes (int): The number of processes
        start_epsilon (float): max value of epsilon
        end_epsilon (float): min value of epsilon
        decay_steps (int): how many steps it takes for epsilon to decay
        gpu (int): GPU device id if not None nor negative.
        update_steps (int): The number of update steps
        pi_loss_coef (float): Weight coefficient for the loss of the policy
        v_loss_coef (float): Weight coefficient for the loss of the value
            function
        entropy_coeff (float): Weight coefficient for the loss of the entropy
        act_deterministically (bool): If set true, choose most probable actions
            in act method.
        max_grad_norm (float or None): Maximum L2 norm of the gradient used for
            gradient clipping. If set to None, the gradient is not clipped.
        average_actor_loss_decay (float): Decay rate of average actor loss.
            Used only to record statistics.
        average_entropy_decay (float): Decay rate of average entropy. Used only
            to record statistics.
        average_value_decay (float): Decay rate of average value. Used only
            to record statistics.
        batch_states (callable): method which makes a batch of observations.
            default is `pfrl.utils.batch_states.batch_states`
    """

    process_idx = None
    saved_attributes = ("model", "optimizer")

    def __init__(
        self,
        model,
        optimizer,
        gamma,
        num_processes,
        # NOTE: additional args for epsilon-greedy
        start_epsilon,
        end_epsilon,
        decay_steps,
        # --------------
        gpu=None,
        update_steps=5,
        pi_loss_coef=1.0,
        v_loss_coef=0.5,
        entropy_coeff=0.01,
        act_deterministically=False,
        max_grad_norm=None,
        average_actor_loss_decay=0.999,
        average_entropy_decay=0.999,
        average_value_decay=0.999,
        batch_states=batch_states,
    ):

        self.model = model
        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.optimizer = optimizer

        self.update_steps = update_steps
        self.num_processes = num_processes

        self.gamma = gamma
        self.act_deterministically = act_deterministically
        self.max_grad_norm = max_grad_norm
        self.phi = lambda x: x
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.entropy_coeff = entropy_coeff

        self.average_actor_loss_decay = average_actor_loss_decay
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        self.batch_states = batch_states

        self.t = 0
        self.t_start = 0

        # Stats
        self.average_actor_loss = 0
        self.average_value = 0
        self.average_entropy = 0

        # NOTE: add epsilon-greedy
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps

    def _flush_storage(self, obs_shape, action):
        obs_shape = obs_shape[1:]
        action_shape = action.shape[1:]

        # NOTE: Increase capacity from update_steps to update_steps * 2
        self.states = torch.zeros(
            self.update_steps * 2 + 1,
            self.num_processes,
            *obs_shape,
            device=self.device,
            dtype=torch.float
        )
        self.actions = torch.zeros(
            self.update_steps * 2,
            self.num_processes,
            *action_shape,
            device=self.device,
            dtype=torch.float
        )
        self.rewards = torch.zeros(
            self.update_steps * 2, self.num_processes, device=self.device, dtype=torch.float
        )
        self.value_preds = torch.zeros(
            self.update_steps * 2 + 1,
            self.num_processes,
            device=self.device,
            dtype=torch.float,
        )

        self.obs_shape = obs_shape
        self.action_shape = action_shape

    def _compute_returns(self):
        """
        """
        _rewards = self.rewards.cpu().numpy()
        _rewards = np.append(_rewards, np.zeros((1, self.num_processes)), axis=0)
        discounted_rewards = discounted_cumulative_rewards(_rewards, self.gamma)[:-1]
        batch_disrewards = discounted_rewards[:self.update_steps] \
            - (self.gamma ** self.update_steps) * discounted_rewards[self.update_steps:]

        self.returns = torch.from_numpy(batch_disrewards).to(self.device)

    def update(self):
        self._compute_returns()

        pout, values = self.model(self.states[:self.update_steps].reshape(-1, *self.obs_shape))

        actions = self.actions[:self.update_steps].reshape(-1, *self.action_shape)
        dist_entropy = pout.entropy().mean()
        action_log_probs = pout.log_prob(actions)

        values = values.reshape((self.update_steps, self.num_processes))
        action_log_probs = action_log_probs.reshape(
            (self.update_steps, self.num_processes)
        )
        advantages = self.returns - values
        value_loss = (advantages * advantages).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()

        (
            value_loss * self.v_loss_coef
            + action_loss * self.pi_loss_coef
            - dist_entropy * self.entropy_coeff
        ).backward()

        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # NOTE: Update time-step
        self.t_start += self.update_steps
        # sliding window
        self.states[:self.update_steps] = self.states[self.update_steps:-1]
        self.actions[:self.update_steps] = self.actions[self.update_steps:]
        self.rewards[:self.update_steps] = self.rewards[self.update_steps:]
        self.value_preds[:self.update_steps] = self.value_preds[self.update_steps:-1]

        # Update stats
        self.average_actor_loss += (1 - self.average_actor_loss_decay) * (
            float(action_loss) - self.average_actor_loss
        )
        self.average_value += (1 - self.average_value_decay) * (
            float(value_loss) - self.average_value
        )
        self.average_entropy += (1 - self.average_entropy_decay) * (
            float(dist_entropy) - self.average_entropy
        )


    def compute_epsilon(self, t):
        # NOTE: that additional function
        if t > self.decay_steps:
            return self.end_epsilon
        else:
            epsilon_diff = self.end_epsilon - self.start_epsilon
            return self.start_epsilon + epsilon_diff * (t / self.decay_steps)


    def _batch_act_train(self, batch_obs):
        assert self.training

        statevar = self.batch_states(batch_obs, self.device, self.phi)

        if self.t == 0:
            with torch.no_grad():
                pout, _ = self.model(statevar)
                action = pout.sample()
            self._flush_storage(statevar.shape, action)

        self.states[self.t - self.t_start] = statevar

        # NOTE: 
        if self.t - self.t_start == self.update_steps * 2:
            self.update()

        # NOTE: inference during training
        with torch.no_grad():
            pout, value = self.model(statevar)
            # epsilon-greedy
            epsilon = self.compute_epsilon(self.t)
            if np.random.rand() < epsilon:
                # random
                action = pout.sample()
            else:
                # greedy
                action = mode_of_distribution(pout)

        self.actions[self.t - self.t_start] = action.reshape(-1, *self.action_shape)
        self.value_preds[self.t - self.t_start] = value[:, 0]

        return action.cpu().numpy()


    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):
        assert self.training

        self.t += 1

        if any(batch_reset):
            warnings.warn(
                "A2C currently does not support resetting an env without reaching a"
                " terminal state during training. When receiving True in batch_reset,"
                " A2C considers it as True in batch_done instead."
            )  # NOQA
            batch_done = list(batch_done)
            for i, reset in enumerate(batch_reset):
                if reset:
                    batch_done[i] = True

        statevar = self.batch_states(batch_obs, self.device, self.phi)

        self.rewards[self.t - self.t_start - 1] = torch.as_tensor(
            batch_reward, device=self.device, dtype=torch.float
        )
        self.states[self.t - self.t_start] = statevar

        # NOTE: 
        if self.t - self.t_start == self.update_steps * 2:
            self.update()

