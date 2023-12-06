from main.model import AtariNet
from main.agent import Agent
from main.environment import *
import torch
import os

# This is the testing script for evaluating the agent's performance in a game

if __name__ == "__main__":
    """
    Testing script for evaluating the trained agent's performance in a game.

    Steps:
    1. Load the required environment, neural network model, and the DQN agent.
    2. Configure the testing parameters such as epsilon value for exploration.
    3. Evaluate the agent's performance in the environment for a specified number of games.

    Usage:
    Run this script to evaluate the trained agent's performance in a game using DQN.

    Note:
    Adjust the epsilon value, games_amount, and other hyperparameters based on the game to be tested.
    """

    # Set environment variable to prevent KMP library error
    os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

    # Check available device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the game environment
    environment = DQNSpaceInvaders(device = device)  # Change environment as needed

    # Create an AtariNet model for the specified game
    model = AtariNet(nb_actions = 6)  # Update nb_actions based on the game's action space

    model.to(device)

    # Load trained model weights
    model.load_the_model()

    # Initialize the agent with specified hyperparameters
    agent = Agent(model = model,
                  device = device,
                  epsilon = 1, # Change epsilon based on where you are in the Exploration, Exploitation trade-off
                  min_epsilon = 0.1,
                  nb_warmup = 3000,
                  nb_actions = 6,  # Update the number of actions for the specific game
                  learning_rate = 0.00025,
                  memory_capacity = 25000,
                  batch_size = 32)

    # Test the agent's performance in the environment for a specified number of games
    # Adjust games_amount based on how many games you want to see in the video in the "videos" folder
    agent.test(env = environment, games_amount = 1)
