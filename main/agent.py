from main.plot import LivePlot
import random
import torch
import copy
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import imageio

class ReplayMemory:
    """
    ReplayMemory: A class representing the experience replay memory for an agent.

    Args:
    - capacity (int): Maximum capacity of the memory buffer.
    - device (str): Device to use for computation ('cpu' or 'cuda').

    Attributes not listed in Args:
    - memory (list): List storing transitions in the memory buffer.
    - position (int): Current position in the memory buffer.

    Methods:
    - insert(transition): Inserts a transition into the replay memory.
    - sample(batch_size): Samples a batch of transitions from the memory.
    - can_sample(batch_size): Checks if enough transitions are available to sample.
    - __len__(): Returns the current length of the memory buffer.
    """

    def __init__(self, capacity, device = "cpu"):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device

    def insert(self, transition):
        """
        Inserts a transition into the replay memory.

        Args:
        - transition (tuple): A tuple containing the elements of the transition (state, action, reward, done, next_state).
        """

        transition = [item.to('cpu') for item in transition]

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory.remove(self.memory[0])
            self.memory.append(transition)

    def sample(self, batch_size = 32):
        """
        Samples a batch of transitions from the memory.

        Args:
        - batch_size (int): Number of transitions to sample.

        Returns:
        - A list containing tensors of transitions: [states, actions, rewards, dones, next_states]
        """

        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        
        return [torch.cat(items).to(self.device) for items in batch]
    
    def can_sample(self, batch_size):
        """
        Checks if enough transitions are available to sample.

        Args:
        - batch_size (int): Number of transitions to sample.

        Returns:
        - True if enough transitions are available to sample a batch of the specified size, otherwise False.
        """

        return len(self.memory) >= batch_size * 10
    
    def __len__(self):
        """
        Returns the current length of the memory buffer.

        Returns:
        - The current number of stored transitions in the memory.
        """

        return len(self.memory)


class Agent:
    """
    Agent: A class representing an agent that interacts with the environment using DQN.

    Args:
    - model (torch.nn.Module): Neural network model for the agent.
    - device (str): Device to use for computation ('cpu' or 'cuda').
    - epsilon (float): Initial value of exploration rate (epsilon-greedy).
    - min_epsilon (float): Minimum value of exploration rate.
    - nb_warmup (int): Number of steps for exploration rate decay.
    - nb_actions (int): Number of possible actions in the environment.
    - memory_capacity (int): Capacity of the experience replay memory.
    - batch_size (int): Batch size for training.
    - learning_rate (float): Learning rate for the optimizer.

    Methods:
    - get_action(state): Chooses an action based on the epsilon-greedy policy.
    - train(env, epochs): Trains the agent using the provided environment for a given number of epochs.
    - test(env, games_amount): Evaluates the trained agent in the environment for a specified number of games.
    """
    
    def __init__(self, model, device, epsilon, min_epsilon, nb_warmup,
                 nb_actions, memory_capacity, batch_size, learning_rate) -> None:
        
        self.memory = ReplayMemory(device = device, capacity = memory_capacity)
        self.model = model
        self.target_model = copy.deepcopy(model).eval()
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = 1 - (((epsilon - min_epsilon) / nb_warmup) * 2)
        self.batch_size = batch_size
        self.model.to(device)
        self.target_model.to(device)
        self.gamma = 0.99
        self.nb_actions = nb_actions

        self.optimizer = optim.Adam(model.parameters(), lr = learning_rate)

        print(f"Epsilon decay is {self.epsilon_decay}")

    def get_action(self, state):
        """
        Chooses an action based on the epsilon-greedy policy.

        Args:
        - state: Current state of the environment.

        Returns:
        - A tensor representing the chosen action.
        """
        
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.nb_actions, (1, 1))
        else:
            av = self.model(state).detach()
            return torch.argmax(av, dim = 1, keepdim = True)
        
    def train(self, env, epochs):
        """
        Trains the agent using the provided environment for a given number of epochs. This is the
        main training loop that is called from the train.py file.

        Args:
        - env (gym.Env): Gym environment.
        - epochs (int): Number of epochs to train the agent.

        Returns:
        - Dictionary containing training statistics: {"Returns": [], "AvgReturns": [], "EpsilonCheckpoints": []}
        """

        stats = {"Returns": [], "AvgReturns": [], "EpsilonCheckpoints": []}

        plotter = LivePlot()

        for epoch in range(1, epochs + 1):
            state = env.reset()
            done = False
            ep_return = 0

            while not done:
                action = self.get_action(state)

                next_state, reward, done, info = env.step(action)

                self.memory.insert([state, action, reward, done, next_state])

                if self.memory.can_sample(self.batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = self.memory.sample(self.batch_size)
                    qsa_b = self.model(state_b).gather(1, action_b)
                    next_qsa_b = self.target_model(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b, dim = 1, keepdim = True)[0]
                    target_b = reward_b + ~done_b * self.gamma * next_qsa_b
                    loss = F.mse_loss(qsa_b, target_b)
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                state = next_state
                ep_return += reward.item()

            stats["Returns"].append(ep_return)

            if self.epsilon > self.min_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay

            if epoch % 20 == 0:
                self.model.save_the_model()
                print(" ")

                average_returns = np.mean(stats["Returns"][-100:])

                stats["AvgReturns"].append(average_returns)
                stats["EpsilonCheckpoints"].append(self.epsilon)

                if (len(stats["Returns"])) > 100:
                    print(f"Epoch: {epoch} - Average Return: {np.mean(stats['Returns'][-100:])} - Epsilon: {self.epsilon}")
                
                else:
                    print(f"Epoch: {epoch} - Episode Return: {np.mean(stats['Returns'][-1:])} - Epsilon: {self.epsilon}")

            if epoch % 50 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            if epoch % 100 == 0:
                plotter.update_plot(stats)

            if epoch % 1000 == 0:
                self.model.save_the_model(f"models/model_iter_{epoch}.pt")

        return stats
    

    def test(self, env, games_amount):
        """
        Evaluates the trained agent in the environment for a specified number of games.

        Args:
        - env (gym.Env): Gym environment for testing.
        - games_amount (int): Number of games to play for evaluation.

        Returns:
        - None (Writes the rendered frames to a video file for evaluation).
        """

        writer = imageio.get_writer('./videos/game_video_test.mp4', fps = 30)

        for epoch in range(games_amount):
            state = env.reset()

            done = False

            for _ in range(1000):
                action = self.get_action(state)
                state, reward, done, info = env.step(action)
                frame = env.render('rgb_array')
                writer.append_data(frame)
                if done:
                    break
        
        writer.close()
        env.close()