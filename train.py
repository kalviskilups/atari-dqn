from main.model import AtariNet
from main.agent import Agent
from main.environment import *
import os
import torch

# This is the training script (adjust environment and hyperparameters
# based on what game you want to play)

if __name__ == "__main__":
    """
    Main training script for training an agent to play a game using Deep Q-Learning (DQN).

    Steps:
    1. Set the appropriate environment and hyperparameters for the game.
    2. Initialize the environment, neural network model, and the DQN agent.
    3. Train the agent using the specified environment and hyperparameters.

    Usage:
    Run this script to train an agent for playing a game using DQN.

    Note:
    Adjust the environment and hyperparameters based on the game to be played.
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

    # Load pre-trained model weights if available
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

    # Train the agent using the specified environment and epochs
    agent.train(env=environment, epochs = 5000)  # Adjust epochs based on training duration
