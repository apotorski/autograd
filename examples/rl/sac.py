import numpy as np
from environment import Environment
from matplotlib import animation, patheffects
from matplotlib import pyplot as plt
from matplotlib._enums import CapStyle, JoinStyle

from autograd import (clip, concatenate, exp, gaussian_log_probability,
                      gaussian_sample, mean_squared_error_loss, minimum,
                      softplus, tanh, tensor)
from autograd.optimizer import RMSProp

plt.style.use('bmh')
plt.rcParams.update({"text.usetex": True})

INITIAL_EXPLORATION_STEPS = 10000

EPOCHS = 100
STEPS_PER_EPOCH = 4000

EVALUATION_TRAJECTORIES = 16
UPDATE_STEPS = 128
BATCH_SIZE = 128

FPS = 25


class Agent:

    MIN_LOG_STANDARD_DEVIATION = -20
    MAX_LOG_STANDARD_DEVIATION = 2

    class Linear:

        def __init__(self, input_features, output_features):
            # Xavier initialization
            bound = 1.0 / np.sqrt(input_features)
            self._W = tensor(np.random.uniform(
                -bound, bound, (output_features, input_features)))
            self._b = tensor(np.zeros((output_features, 1)))

        def __call__(self, x):
            return self._W @ x + self._b

        @property
        def parameters(self):
            return self._W, self._b

    class Actor:

        def __init__(self, state_space, action_space):
            self._input_embedder = Agent.Linear(state_space, 64)
            self._middleware = Agent.Linear(64, 64)
            self._action_mean_head = Agent.Linear(64, action_space)
            self._log_action_standard_deviation_head = \
                Agent.Linear(64, action_space)

        def __call__(self, observation):
            x = tanh(self._input_embedder(observation))
            x = tanh(self._middleware(x))
            action_mean = self._action_mean_head(x)
            log_action_standard_deviation = \
                self._log_action_standard_deviation_head(x)

            return action_mean, log_action_standard_deviation

        @property
        def parameters(self):
            return self._input_embedder.parameters \
                + self._middleware.parameters \
                + self._action_mean_head.parameters \
                + self._log_action_standard_deviation_head.parameters

        def save(self, path):
            with open(path, 'wb') as file:
                values = tuple(parameter.data for parameter in self.parameters)
                np.savez(file, *values)

        def load(self, path):
            with open(path, 'rb') as file:
                for parameter, value in zip(self.parameters,
                                            np.load(file).values()):
                    parameter.data = value

    class Critic:

        def __init__(self, state_space, action_space):
            self._input_embedder = Agent.Linear(state_space + action_space, 64)
            self._middleware = Agent.Linear(64, 64)
            self._action_value_head = Agent.Linear(64, 1)

        def __call__(self, observation, action):
            x = concatenate((observation, action), axis=-2)
            x = tanh(self._input_embedder(x))
            x = tanh(self._middleware(x))
            x = self._action_value_head(x)

            return x

        @property
        def parameters(self):
            return self._input_embedder.parameters \
                + self._middleware.parameters \
                + self._action_value_head.parameters

        def save(self, path):
            with open(path, 'wb') as file:
                values = tuple(parameter.data for parameter in self.parameters)
                np.savez(file, *values)

        def load(self, path):
            with open(path, 'rb') as file:
                for parameter, value in zip(self.parameters,
                                            np.load(file).values()):
                    parameter.data = value

    def __init__(self, observation_space, action_space,
                 buffer_capacity=1000000, batch_size=128, learning_rate=1e-3,
                 gamma=0.99, tau=0.005):
        self._buffer_capacity = buffer_capacity
        self._batch_size = batch_size

        self._buffer_indice = 0
        self._buffer_size = 0

        self._observation_buffer = np.empty(
            (self._buffer_capacity, observation_space, 1), dtype=np.float32)
        self._action_buffer = np.empty(
            (self._buffer_capacity, action_space, 1), dtype=np.float32)
        self._reward_buffer = np.empty(
            (self._buffer_capacity, 1, 1), dtype=np.float32)
        self._next_observation_buffer = np.empty(
            (self._buffer_capacity, observation_space, 1), dtype=np.float32)
        self._not_done_buffer = np.empty(
            (self._buffer_capacity, 1, 1), dtype=bool)


        self._actor = Agent.Actor(observation_space, action_space)

        self._actor_optimizer = RMSProp(
            self._actor.parameters, learning_rate=learning_rate,
            weight_decay=1e-3)

        self._critic_1 = Agent.Critic(observation_space, action_space)
        self._critic_2 = Agent.Critic(observation_space, action_space)

        self._critic_optimizer = RMSProp(
            self._critic_1.parameters + self._critic_2.parameters,
            learning_rate=learning_rate, weight_decay=1e-3)

        self._target_critic_1 = Agent.Critic(observation_space, action_space)
        for target_parameter, parameter \
                in zip(self._target_critic_1.parameters,
                    self._critic_1.parameters):
            target_parameter.data = parameter.data.copy()

        self._target_critic_2 = Agent.Critic(observation_space, action_space)
        for target_parameter, parameter \
                in zip(self._target_critic_2.parameters,
                    self._critic_2.parameters):
            target_parameter.data = parameter.data.copy()

        self._log_alpha = tensor(0.0, requires_gradient=True)

        self._alpha_optimizer = RMSProp((self._log_alpha,),
                                        learning_rate=learning_rate)

        self._alpha = exp(self._log_alpha).detach()

        self._target_entropy = tensor(-action_space, requires_gradient=False)

        self._gamma = tensor(gamma, requires_gradient=False)

        self._tau = tau

    def record(self, observation, action, reward, next_observation, done):
        self._observation_buffer[self._buffer_indice] = observation
        self._action_buffer[self._buffer_indice] = action
        self._reward_buffer[self._buffer_indice] = reward
        self._next_observation_buffer[self._buffer_indice] = next_observation
        self._not_done_buffer[self._buffer_indice] = not(done)

        self._buffer_indice = (self._buffer_indice + 1) % self._buffer_capacity

        self._buffer_size = self._buffer_size + 1 \
            if self._buffer_size < self._buffer_capacity else \
                self._buffer_capacity

    def explore(self, state):
        action_mean, log_action_standard_deviation = self._actor(state)

        log_action_standard_deviation = clip(log_action_standard_deviation,
                                             Agent.MIN_LOG_STANDARD_DEVIATION,
                                             Agent.MAX_LOG_STANDARD_DEVIATION)
        action_standard_deviation = exp(log_action_standard_deviation)

        action = gaussian_sample(action_mean, action_standard_deviation)
        action = tanh(action)

        return action.data

    def exploit(self, state):
        action_mean, _ = self._actor(state)
        action = tanh(action_mean)

        return action.data

    def _reparameterize(self, action_mean, log_action_standard_deviation):
        log_action_standard_deviation = clip(log_action_standard_deviation,
                                             Agent.MIN_LOG_STANDARD_DEVIATION,
                                             Agent.MAX_LOG_STANDARD_DEVIATION)
        action_standard_deviation = exp(log_action_standard_deviation)
        action = gaussian_sample(action_mean, action_standard_deviation)

        # more numerically stable equivalent of equation 26.
        # (https://arxiv.org/abs/1812.05905)
        log_action_probability = \
            gaussian_log_probability(
                action, action_mean, action_standard_deviation) \
                    + (2.0*(action + softplus(-2.0*action) - np.log(2.0))) \
                        .sum(axis=-2, keep_dimensions=True)

        action = tanh(action)

        return log_action_probability, action

    def _train_step(self, observation, action, reward, next_observation,
                    not_done):
        # compute targets for action-value functions
        next_log_action_probability, next_action = \
            self._reparameterize(*self._actor(next_observation))

        next_action_value_1 = self._target_critic_1(
            next_observation, next_action)
        next_action_value_2 = self._target_critic_2(
            next_observation, next_action)
        next_action_value = minimum(next_action_value_1, next_action_value_2)

        action_value = reward + self._gamma*(next_action_value
            - self._alpha*next_log_action_probability)*not_done


        # update action-value functions
        predicted_action_value_1 = self._critic_1(observation, action)
        predicted_action_value_2 = self._critic_2(observation, action)

        critic_loss_1 = mean_squared_error_loss(predicted_action_value_1,
                                                action_value.detach())
        critic_loss_2 = mean_squared_error_loss(predicted_action_value_2,
                                                action_value.detach())

        critic_loss = critic_loss_1 + critic_loss_2

        critic_loss.backpropagate()
        self._critic_optimizer.step()


        # update policy
        log_action_probability, action = \
            self._reparameterize(*self._actor(observation))

        action_value_1 = self._critic_1(observation, action)
        action_value_2 = self._critic_2(observation, action)
        action_value = minimum(action_value_1, action_value_2)

        actor_loss = (self._alpha*log_action_probability - action_value).mean()

        actor_loss.backpropagate()
        self._actor_optimizer.step()


        # update entropy factor
        alpha_loss = (-self._log_alpha*(log_action_probability.detach()
            + self._target_entropy)).mean()

        alpha_loss.backpropagate()
        self._alpha_optimizer.step()

        self._alpha = exp(self._log_alpha).detach()


        # update target networks
        for target_parameter, parameter \
                in zip(self._target_critic_1.parameters,
                    self._critic_1.parameters):
            target_parameter.data = target_parameter.data*(1.0 - self._tau) \
                + self._tau*parameter.data

        for target_parameter, parameter \
                in zip(self._target_critic_2.parameters,
                    self._critic_2.parameters):
            target_parameter.data = target_parameter.data*(1.0 - self._tau) \
                + self._tau*parameter.data

    def train(self):
        for _ in range(UPDATE_STEPS):
            indices = np.random.choice(self._buffer_size, self._batch_size,
                                       replace=False)

            observation = tensor(self._observation_buffer[indices],
                                 requires_gradient=False)
            action = tensor(self._action_buffer[indices],
                            requires_gradient=False)
            reward = tensor(self._reward_buffer[indices],
                            requires_gradient=False)
            next_observation = tensor(self._next_observation_buffer[indices],
                                      requires_gradient=False)
            not_done = tensor(self._not_done_buffer[indices],
                              requires_gradient=False)

            self._train_step(observation, action, reward, next_observation,
                             not_done)

    @property
    def actor(self):
        return self._actor

    @property
    def critic_1(self):
        return self._critic_1

    @property
    def critic_2(self):
        return self._critic_2


if __name__ == '__main__':
    environment = Environment()

    agent = Agent(environment.observation_space, environment.action_space,
                  batch_size=BATCH_SIZE)

    # train agent
    observation = environment.reset()
    for _ in range(INITIAL_EXPLORATION_STEPS):
        action = np.random.uniform(-1.0, 1.0)
        next_observation, reward, done = environment(action)
        agent.record(observation, action, reward, next_observation, done)
        observation = environment.reset() if done else next_observation

    for epoch in range(EPOCHS):
        observation = environment.reset()
        for _ in range(STEPS_PER_EPOCH):
            action = agent.explore(observation)
            next_observation, reward, done = environment(action)
            agent.record(observation, action, reward, next_observation, done)
            observation = environment.reset() if done else next_observation

        agent.train()


        total_return = 0.0
        for _ in range(EVALUATION_TRAJECTORIES):
            observation = environment.reset()
            while True:
                action = agent.exploit(observation)
                observation, reward, done = environment(action)
                total_return += float(reward)

                if done:
                    break

        average_return = total_return / EVALUATION_TRAJECTORIES

        print(f'epoch: {epoch + 1} / {EPOCHS}, '
              f'average return = {average_return:.3f}')

    agent.actor.save('./models/sac/actor.npz')
    agent.critic_1.save('./models/sac/critic_1.npz')
    agent.critic_2.save('./models/sac/critic_2.npz')

    # plot policy and value function
    theta = np.arange(0.0, 2.0*np.pi, 1e-2)
    dot_theta = np.arange(-2.0, 2.0, 1e-2)

    Theta, Dot_Theta = np.meshgrid(theta, dot_theta)
    observations = np.vstack((Theta.flatten(), Dot_Theta.flatten())) \
        .transpose().reshape(-1, 2, 1)
    actions = agent.exploit(observations)
    Actions = actions.reshape(Theta.shape)

    Values = minimum(agent.critic_1(observations, actions),
                     agent.critic_2(observations, actions)).data \
                        .reshape(Theta.shape)

    px = 1.0 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(640*px, 640*px))
    ax.set_title('Policy (SAC)')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\dot{\theta}$')
    cs_0 = ax.contourf(Theta, Dot_Theta, Actions)
    cbar_0 = fig.colorbar(cs_0, ax=ax)
    cbar_0.ax.set_ylabel(r'$\pi(s)$')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig('sac_policy.png')

    fig, ax = plt.subplots(figsize=(640*px, 640*px))
    ax.set_title('Value function (SAC)')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\dot{\theta}$')
    cs_1 = ax.contourf(Theta, Dot_Theta, Values)
    cbar_1 = fig.colorbar(cs_1, ax=ax)
    cbar_1.ax.set_ylabel(r'$V(s)$')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig('sac_value_function.png')

    # create animation
    pendulum_angles = []

    initial_state = np.array([
        [np.pi * 5.0 / 4.0],
        [0.0]
    ])
    observation = environment.reset(initial_state)
    while True:
        pendulum_angles.append(observation[0, 0])
        action = agent.exploit(observation)
        observation, _, done = environment(action)

        if done:
            break

    fig, ax = plt.subplots(figsize=(640*px, 640*px))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect('equal')

    x = (0.0, environment.l*np.cos(pendulum_angles[0] + np.pi / 2.0))
    y = (0.0, environment.l*np.sin(pendulum_angles[0] + np.pi / 2.0))
    pendulum, = ax.plot(
        x, y, color='tab:blue', lw=10,
        path_effects=[patheffects.SimpleLineShadow(), patheffects.Normal()],
        solid_capstyle=CapStyle.round, solid_joinstyle=JoinStyle.round)

    fig.tight_layout()

    def animate(pendulum_angle):
        x = (0.0, environment.l*np.cos(pendulum_angle + np.pi / 2.0))
        y = (0.0, environment.l*np.sin(pendulum_angle + np.pi / 2.0))
        pendulum.set_data(x, y)

        return pendulum,

    animation.FuncAnimation(
        fig, animate, pendulum_angles, interval=int(1e3 / FPS), blit=True) \
            .save('sac_demo.gif', fps=FPS)