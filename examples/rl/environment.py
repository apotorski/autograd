import numpy as np


class Environment:

    _g = 9.807

    def __init__(self, m=0.5, l=1.0, b=1e-2, dt=0.04):
        self._dt = dt

        self._m = m
        self._l = l
        self._b = b

        self._max_u = 2.0
        self._max_theta_dot = 8.0

        self._max_episode_length = 200

        self._t = 0

        theta = np.random.uniform(-np.pi, np.pi)
        theta_dot = np.random.uniform(-1.0, 1.0)
        self._state = np.array([
            [theta],
            [theta_dot]
        ])

    def reset(self, initial_state=None):
        self._t = 0

        if initial_state is None:
            theta = np.random.uniform(-np.pi, np.pi)
            theta_dot = np.random.uniform(-1.0, 1.0)
            self._state = np.array([
                [theta],
                [theta_dot]
            ])
        else:
            theta, theta_dot = np.squeeze(initial_state)
            theta = ((theta + np.pi) % (2.0*np.pi)) - np.pi
            theta_dot = np.clip(theta_dot, -self._max_theta_dot,
                                            self._max_theta_dot)
            self._state = np.array([
                [theta],
                [theta_dot]
            ])

        return self._state.copy()

    def __call__(self, action):
        theta, theta_dot = np.squeeze(self._state)

        u = np.squeeze(action)
        u = np.clip(u, -1.0, 1.0)*self._max_u

        theta_dot += 3.0*(np.sin(theta)*self._g/self._l
            + (u - self._b*theta_dot)/(self._m*self._l**2.0)) * self._dt
        theta_dot = np.clip(theta_dot, -self._max_theta_dot,
                                        self._max_theta_dot)

        theta += theta_dot * self._dt
        theta = ((theta + np.pi) % (2.0*np.pi)) - np.pi

        self._state = np.array([
            [theta],
            [theta_dot]
        ])

        reward = -(theta**2.0 + 1e-1*theta_dot**2.0 + 1e-2*u**2.0)

        self._t += 1
        done_flag = self._t >= self._max_episode_length

        return self._state.copy(), reward, done_flag

    @property
    def observation_space(self):
        return 2

    @property
    def action_space(self):
        return 1

    @property
    def l(self):
        return self._l