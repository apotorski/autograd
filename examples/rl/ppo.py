import numpy as np
from environment import Environment
from matplotlib import animation, patheffects
from matplotlib import pyplot as plt
from matplotlib._enums import CapStyle, JoinStyle

from autograd import (clip, exp, gaussian_entropy, gaussian_log_probability,
                      gaussian_sample, mean_squared_error_loss, minimum, tanh,
                      tensor)
from autograd.optimizer import RMSProp

plt.style.use('bmh')
plt.rcParams.update({"text.usetex": True})


EPOCHS = 100
STEPS_PER_EPOCH = 4000

EVALUATION_TRAJECTORIES = 16

GAMMA = 0.99
LAMBDA = 0.97
EPSILON = 0.2
ENTROPY_COEFFICIENT = 1e-2

UPDATE_STEPS = 64
BATCH_SIZE = 128

FPS = 25


class Agent:

    class Linear:

        def __init__(self, input_features, output_features):
            # Xavier initialization
            bound = 1.0 / np.sqrt(input_features)
            self._W = tensor(
                np.random.uniform(-bound, bound,
                    (output_features, input_features)))
            self._b = tensor(np.zeros((output_features, 1)))

        def __call__(self, x):
            return self._W @ x + self._b

        @property
        def parameters(self):
            return self._W, self._b

    class Policy:

        MIN_LOG_STANDARD_DEVIATION = -20
        MAX_LOG_STANDARD_DEVIATION = 2

        def __init__(self, observation_space, action_space):
            self._input_embedder = Agent.Linear(observation_space, 64)
            self._middleware = Agent.Linear(64, 64)
            self._action_mean_head = Agent.Linear(64, 1)

            self._log_action_standard_deviation = tensor(
                -0.5*np.ones((action_space, 1)))

        def __call__(self, observation):
            h = tanh(self._input_embedder(observation))
            h = tanh(self._middleware(h))
            action_mean = self._action_mean_head(h)

            log_action_standard_deviation = clip(
                self._log_action_standard_deviation,
                Agent.Policy.MIN_LOG_STANDARD_DEVIATION,
                Agent.Policy.MAX_LOG_STANDARD_DEVIATION)
            action_standard_deviation = exp(log_action_standard_deviation)

            return action_mean, action_standard_deviation

        @property
        def parameters(self):
            return self._input_embedder.parameters \
                + self._middleware.parameters \
                + self._action_mean_head.parameters \
                + (self._log_action_standard_deviation,)

        def save(self, path):
            with open(path, 'wb') as file:
                values = tuple(parameter.data for parameter in self.parameters)
                np.savez(file, *values)

        def load(self, path):
            with open(path, 'rb') as file:
                for parameter, value in zip(self.parameters,
                                            np.load(file).values()):
                    parameter.data = value

    class ValueFunction:

        def __init__(self, observation_space):
            self._input_embedder = Agent.Linear(observation_space, 64)
            self._middleware = Agent.Linear(64, 64)
            self._value_head = Agent.Linear(64, 1)

        def __call__(self, observation):
            h = tanh(self._input_embedder(observation))
            h = tanh(self._middleware(h))
            value = self._value_head(h)

            return value

        @property
        def parameters(self):
            return self._input_embedder.parameters \
                + self._middleware.parameters \
                + self._value_head.parameters

        def save(self, path):
            with open(path, 'wb') as file:
                values = tuple(parameter.data for parameter in self.parameters)
                np.savez(file, *values)

        def load(self, path):
            with open(path, 'rb') as file:
                for parameter, value in zip(self.parameters,
                                            np.load(file).values()):
                    parameter.data = value

    def __init__(self, observation_space, action_space, buffer_size,
                 batch_size=128, policy_learning_rate=3e-4,
                 value_function_learning_rate=1e-3):
        self._observations = np.empty((buffer_size, observation_space, 1),
                                      dtype=np.float32)
        self._actions = np.empty((buffer_size, action_space, 1),
                                 dtype=np.float32)
        self._rewards = np.empty((buffer_size, 1, 1),
                                 dtype=np.float32)
        self._not_done = np.empty((buffer_size, 1, 1),
                                         dtype=bool)

        self._buffer_indice = 0
        self._buffer_size = buffer_size

        self._batch_size = batch_size


        self._policy = Agent.Policy(observation_space, action_space)
        self._policy_optimizer = RMSProp(
            self._policy.parameters,
            learning_rate=policy_learning_rate,
            weight_decay=1e-3)

        self._value_function = Agent.ValueFunction(observation_space)
        self._value_function_optimizer = RMSProp(
            self._value_function.parameters,
            learning_rate=value_function_learning_rate,
            weight_decay=1e-3)

    def record(self, observation, action, reward, done):
        if self._buffer_indice < self._buffer_size:
            self._observations[self._buffer_indice] = observation
            self._actions[self._buffer_indice] = action
            self._rewards[self._buffer_indice] = reward
            self._not_done[self._buffer_indice] = not(done)

            self._buffer_indice += 1

    def _train_step(self, observation, action, previous_log_action_probability,
                    reward_to_go, advantage):
        # update policy
        action_mean, action_standard_deviation = self._policy(observation)
        log_action_probability = gaussian_log_probability(
            action, action_mean, action_standard_deviation)

        action_probability_ratio = exp(log_action_probability
            - previous_log_action_probability)

        clipped_action_probability_ratio = clip(action_probability_ratio,
                                                1.0 - EPSILON, 1.0 + EPSILON)

        policy_loss = -minimum(
            action_probability_ratio*advantage,
            clipped_action_probability_ratio*advantage
        ).mean()

        exploration_loss = -gaussian_entropy(action_standard_deviation).mean()

        loss = policy_loss + ENTROPY_COEFFICIENT*exploration_loss

        loss.backpropagate()
        self._policy_optimizer.step()

        # update value function
        value = self._value_function(observation)
        loss = mean_squared_error_loss(value, reward_to_go)

        loss.backpropagate()
        self._value_function_optimizer.step()

    def train(self):
        rewards_to_go = np.empty_like(self._rewards)
        advantages = np.empty_like(self._rewards)

        values = self._value_function(self._observations).data

        rewards_to_go[-1] = self._rewards[-1]
        advantages[-1] = self._rewards[-1] - values[-1]
        for indice in reversed(range(self._buffer_indice - 1)):
            rewards_to_go[indice] = self._rewards[indice] \
                + GAMMA*rewards_to_go[indice + 1]*self._not_done[indice]

            delta = self._rewards[indice] \
                + GAMMA*values[indice + 1]*self._not_done[indice] \
                    - values[indice]
            advantages[indice] = delta \
                + GAMMA*LAMBDA*advantages[indice + 1]*self._not_done[indice]

        action_means, action_standard_deviations = \
            self._policy(self._observations)
        previous_log_action_probabilities = gaussian_log_probability(
            self._actions, action_means, action_standard_deviations).data

        for _ in range(UPDATE_STEPS):
            permuted_indices = np.random.permutation(self._buffer_indice)
            X = (x[permuted_indices] for x in (
                    self._observations, self._actions,
                        previous_log_action_probabilities,
                            rewards_to_go, advantages))
            X = [np.array_split(x, x.shape[0] // self._batch_size) for x in X]

            for observation, action, \
                    previous_log_action_probability, \
                        reward_to_go, advantage in zip(*X):
                self._train_step(observation, action,
                                 previous_log_action_probability,
                                 reward_to_go, advantage)

        self._buffer_indice = 0

    def explore(self, observation):
        action_mean, action_standard_deviation = self._policy(observation)
        action = gaussian_sample(action_mean, action_standard_deviation)

        return action.data

    def exploit(self, observation):
        action_mean, _ = self._policy(observation)

        return action_mean.data

    @property
    def policy(self):
        return self._policy

    @property
    def value_function(self):
        return self._value_function


if __name__ == '__main__':
    environment = Environment()
    agent = Agent(environment.observation_space,
                  environment.action_space,
                  STEPS_PER_EPOCH, BATCH_SIZE)

    # train agent
    for epoch in range(EPOCHS):
        observation = environment.reset()
        for _ in range(STEPS_PER_EPOCH):
            action = agent.explore(observation)
            next_observation, reward, done = environment(action)
            agent.record(observation, action, reward, done)
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

    agent.policy.save('./models/ppo/policy.npz')
    agent.value_function.save('./models/ppo/value_function.npz')

    # plot policy and value function
    theta = np.arange(0.0, 2.0*np.pi, 1e-2)
    dot_theta = np.arange(-2.0, 2.0, 1e-2)
    Theta, Dot_Theta = np.meshgrid(theta, dot_theta)
    observations = np.vstack((Theta.flatten(), Dot_Theta.flatten())) \
        .transpose().reshape(-1, 2, 1)

    Actions = agent.exploit(observations) \
        .reshape(Theta.shape)
    Values = agent.value_function(observations).data \
        .reshape(Theta.shape)

    px = 1.0 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(640*px, 640*px))
    ax.set_title('Policy (PPO)')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\dot{\theta}$')
    cs_0 = ax.contourf(Theta, Dot_Theta, Actions)
    cbar_0 = fig.colorbar(cs_0, ax=ax)
    cbar_0.ax.set_ylabel(r'$\pi(s)$')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig('ppo_policy.png')

    fig, ax = plt.subplots(figsize=(640*px, 640*px))
    ax.set_title('Value function (PPO)')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\dot{\theta}$')
    cs_1 = ax.contourf(Theta, Dot_Theta, Values)
    cbar_1 = fig.colorbar(cs_1, ax=ax)
    cbar_1.ax.set_ylabel(r'$V(s)$')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig('ppo_value_function.png')


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
            .save('ppo_demo.gif', fps=FPS)