import numpy as np
import jax
from jax import numpy as jnp
from jax import random as jrandom
from flax import linen as nn
from flax import optim
from flax import serialization

import utils


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):

    action_dim: int
    max_action: float

    def setup(self):
        self.l1 = nn.Dense(256)
        self.l2 = nn.Dense(256)
        self.l3 = nn.Dense(self.action_dim)

    def __call__(self, state):
        a = nn.relu(self.l1(state))
        a = nn.relu(self.l2(a))
        action = self.max_action * nn.tanh(self.l3(a))
        return action


class Critic(nn.Module):

    def setup(self):
        # Q1 architecture
        self.l1 = nn.Dense(256)
        self.l2 = nn.Dense(256)
        self.l3 = nn.Dense(1)

        # Q2 architecture
        self.l4 = nn.Dense(256)
        self.l5 = nn.Dense(256)
        self.l6 = nn.Dense(1)

    def __call__(self, state, action):
        sa = jnp.concatenate([state, action], axis=-1)

        q1 = nn.relu(self.l1(sa))
        q1 = nn.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = nn.relu(self.l4(sa))
        q2 = nn.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = jnp.concatenate([state, action], axis=-1)

        q1 = nn.relu(self.l1(sa))
        q1 = nn.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3:

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        seed=0,
        jit=False,
    ):  

        self.rngs = utils.PRNGKeys(seed)

        # initialize models
        dummy_state = jnp.ones([1, state_dim], dtype=jnp.float32)
        dummy_action = jnp.ones([1, action_dim], dtype=jnp.float32)

        actor_rng = self.rngs.get_key()
        self.actor_model = Actor(action_dim, max_action)
        self.actor_params = self.actor_model.init(actor_rng, dummy_state)
        self.actor_target_params = self.actor_model.init(actor_rng, dummy_state)
        self.actor_optimizer = optim.Adam(learning_rate=3e-4).create(self.actor_params)

        critic_rng = self.rngs.get_key()
        self.critic_model = Critic()
        self.critic_params = self.critic_model.init(
                critic_rng, dummy_state, dummy_action)
        self.critic_target_params = self.critic_model.init(
                critic_rng, dummy_state, dummy_action) 
        self.critic_optimizer = optim.Adam(learning_rate=3e-4).create(self.critic_params)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        if jit:
            self.actor_step = jax.jit(self._actor_step)
            self.critic_step = jax.jit(self._critic_step)
            self.update_target_params = jax.jit(self._update_target_params)
            self.actor = jax.jit(self._actor)
        else:
            self.actor_step = self._actor_step
            self.critic_step = self._critic_step
            self.update_target_params = self._update_target_params
            self.actor = self._actor

    def _actor(self, actor_params, state):
        return self.actor_model.apply(actor_params, state)

    def select_action(self, state):
        state = jax.device_put(state[None])
        return np.array(self.actor(self.actor_params, state)).flatten()

    def _critic_loss(self,
            critic_params, 
            critic_target_params,
            actor_target_params,
            transition,
            rng):
        state, action, next_state, reward, not_done = transition

        # get next action
        noise = jrandom.normal(rng, action.shape) * self.policy_noise
        noise = jnp.clip(noise, -self.noise_clip, self.noise_clip)
        next_action = self.actor_model.apply(actor_target_params, next_state)
        next_action = jnp.clip(next_action + noise, -self.max_action, self.max_action)
        
        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_model.apply(
                critic_target_params, next_state, next_action)
        target_Q = jnp.minimum(target_Q1, target_Q2)
        target_Q = reward + not_done * self.discount * target_Q
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic_model.apply(critic_params, state, action)

        # Compute critic loss
        Q1_loss = jnp.mean(jnp.square(current_Q1 - target_Q))
        Q2_loss = jnp.mean(jnp.square(current_Q2 - target_Q))
        critic_loss = Q1_loss + Q2_loss

        return critic_loss

    def _critic_step(self,
            critic_optimizer,
            critic_target_params,
            actor_target_params,
            transition,
            rng):
        critic_params = critic_optimizer.target
        vgrad_fn = jax.value_and_grad(self._critic_loss, argnums=0)
        loss, grad = vgrad_fn(
                critic_params,
                critic_target_params,
                actor_target_params,
                transition,
                rng)
        critic_optimizer = critic_optimizer.apply_gradient(grad)
        return critic_optimizer, loss

    def _actor_loss(self, actor_params, critic_params, state):
        action = self.actor_model.apply(actor_params, state)
        q_val = self.critic_model.apply(
                critic_params, state, action, method=self.critic_model.Q1)
        actor_loss = -jnp.mean(q_val)
        return actor_loss

    def _actor_step(self, actor_optimizer, critic_params, state):
        actor_params = actor_optimizer.target
        vgrad_fn = jax.value_and_grad(self._actor_loss, argnums=0)
        loss, grad = vgrad_fn(actor_params, critic_params, state)
        actor_optimizer = actor_optimizer.apply_gradient(grad)
        return actor_optimizer, loss

    def _update_target_params(self, params, target_params):
        def _update(param, target_param):
            return self.tau * param + (1 - self.tau) * target_param
        updated_params = jax.tree_multimap(_update, params, target_params)
        return updated_params

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        transition = (state, action, next_state, reward, not_done)

        # critic step
        critic_step_rng = self.rngs.get_key()
        self.critic_optimizer, _ = self.critic_step(
                self.critic_optimizer,
                self.critic_target_params,
                self.actor_target_params,
                transition,
                critic_step_rng)
        self.critic_params = self.critic_optimizer.target
        

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # actor_step
            self.actor_optimizer, _ = self.actor_step(
                self.actor_optimizer,
                self.critic_params,
                state)
            self.actor_params = self.actor_optimizer.target

            # Update the frozen target models
            params = (self.actor_params, self.critic_params)
            target_params = (self.actor_target_params, self.critic_target_params)
            updated_params = self.update_target_params(params, target_params)
            self.actor_target_params, self.critic_target_params = updated_params

    def save(self, filename):
        critic_file = filename + '_critic.ckpt'
        with open(critic_file, 'wb') as f:
            f.write(serialization.to_bytes(self.critic_params))
        actor_file = filename + '_actor.ckpt'
        with open(actor_file, 'wb') as f:
            f.write(serialization.to_bytes(self.actor_params))

    def load(self, filename):
        critic_file = filename + '_critic.ckpt'
        with open(critic_file, 'rb') as f:
            self.critic_params = serialization.from_bytes(self.critic_params, f.read())
        self.critic_target_params = self.critic_params
        actor_file = filename + '_actor.ckpt'
        with open(actor_file, 'rb') as f:
            self.actor_params = serialization.from_bytes(self.actor_params, f.read())
        self.actor_target_params = self.actor_params
        