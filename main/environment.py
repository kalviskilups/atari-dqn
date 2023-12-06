import gym
import numpy as np
from PIL import Image
import torch

class GymWrapperBase(gym.Wrapper):
    """
    GymWrapperBase: Base wrapper class for Gym environments.

    Args:
    - env_name (str): Name of the Gym environment.
    - render_mode (str): Mode for rendering the environment.
    - repeat (int): Number of times to repeat an action.
    - device (str): Device to use for computation ('cpu' or 'cuda').

    Attributes not listed in Args:
    - repeat (int): Number of times to repeat an action.
    - lives (int): Number of lives in the environment.
    - frame_buffer (list): Buffer to store frames.
    - image_shape (tuple): Shape of the image (height, width).

    Methods:
    - step(action): Executes an action in the environment.
    - process_observation(observation): Preprocesses the observation/frame.
    - reset(): Resets the environment and returns the initial observation.
    """

    def __init__(self, env_name, render_mode = "rgb_array", repeat = 4, device = "cpu"):
        """
        Initializes the GymWrapperBase class.

        Args:
        - env_name (str): Name of the Gym environment.
        - render_mode (str): Mode for rendering the environment.
        - repeat (int): Number of times to repeat an action.
        - device (str): Device to use for computation ('cpu' or 'cuda').
        """

        env = gym.make(env_name, render_mode = render_mode)
        super(GymWrapperBase, self).__init__(env)
        self.repeat = repeat
        self.lives = env.ale.lives()
        self.frame_buffer = []
        self.device = device
        self.image_shape = (84, 84)

    def step(self, action):
        """
        Executes an action in the environment.

        Args:
        - action: Action to be taken in the environment.

        Returns:
        - max_frame (torch.Tensor): Preprocessed observation/frame.
        - total_reward (torch.Tensor): Total reward obtained from the action.
        - done (torch.Tensor): Flag indicating if the episode is done.
        - info (dict): Additional information from the environment.
        """

        total_reward = 0
        done = False

        for _ in range(self.repeat):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward

            current_lives = info["lives"]
            if current_lives < self.lives:
                total_reward = total_reward - 1
                self.lives = current_lives

            self.frame_buffer.append(observation)

            if done:
                break

        max_frame = np.max(self.frame_buffer[-2:], axis=0)
        max_frame = self.process_observation(max_frame)
        max_frame = max_frame.to(self.device)

        total_reward = torch.tensor(total_reward).view(1, -1).float().to(self.device)
        done = torch.tensor(done).view(1, -1).to(self.device)

        return max_frame, total_reward, done, info

    def process_observation(self, observation):
        """
        Preprocesses the observation/frame.

        Args:
        - observation: Raw observation/frame from the environment.

        Returns:
        - img (torch.Tensor): Preprocessed image tensor.
        """

        img = Image.fromarray(observation)
        img = img.resize(self.image_shape)
        img = img.convert("L")
        img = np.array(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.unsqueeze(0)
        img = img / 255.0

        img = img.to(self.device)
        return img

    def reset(self):
        """
        Resets the environment and returns the initial observation.

        Returns:
        - observation (torch.Tensor): Initial preprocessed observation.
        """

        self.frame_buffer = []
        observation = self.env.reset()
        self.lives = self.env.ale.lives()
        observation = self.process_observation(observation)
        return observation

class DQNBreakout(GymWrapperBase):
    """
    DQNBreakout: Wrapper class for the BreakoutNoFrameskip-v4 Gym environment.
    Inherits from GymWrapperBase.

    Args:
    - render_mode (str): Mode for rendering the environment.
    - repeat (int): Number of times to repeat an action.
    - device (str): Device to use for computation ('cpu' or 'cuda').
    """

    def __init__(self, render_mode = "rgb_array", repeat = 4, device="cpu"):
        super(DQNBreakout, self).__init__("BreakoutNoFrameskip-v4", render_mode, repeat, device)

class DQNPong(GymWrapperBase):
    """
    DQNPong: Wrapper class for the Pong-ramNoFrameskip-v4 Gym environment.
    Inherits from GymWrapperBase.

    Args:
    - render_mode (str): Mode for rendering the environment.
    - repeat (int): Number of times to repeat an action.
    - device (str): Device to use for computation ('cpu' or 'cuda').
    """

    def __init__(self, render_mode = "rgb_array", repeat = 4, device="cpu"):
        super(DQNPong, self).__init__("Pong-ramNoFrameskip-v4", render_mode, repeat, device)

class DQNSpaceInvaders(GymWrapperBase):
    """
    DQNSpaceInvaders: Wrapper class for the SpaceInvaders-ramNoFrameskip-v4 Gym environment.
    Inherits from GymWrapperBase.

    Args:
    - render_mode (str): Mode for rendering the environment.
    - repeat (int): Number of times to repeat an action.
    - device (str): Device to use for computation ('cpu' or 'cuda').
    """

    def __init__(self, render_mode = "rgb_array", repeat = 3, device="cpu"):
        super(DQNSpaceInvaders, self).__init__("SpaceInvaders-ramNoFrameskip-v4", render_mode, repeat, device)
