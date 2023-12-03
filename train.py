from main.model import AtariNet
from main.agent import Agent
from main.environment import *
import os
import torch

# This is the training script (adjust environment and hyperparameters
# based on what game you want to play)

if __name__ == "__main__":

    os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    environment = DQNPong(device = device)

    model = AtariNet(nb_actions = 6)

    model.to(device)

    #model.load_the_model()

    agent = Agent(model = model, 
                device = device,
                epsilon = 1,
                min_epsilon = 0.1,
                nb_warmup = 2000,
                nb_actions = 6,
                learning_rate = 0.00025,
                memory_capacity = 80000,
                batch_size = 32)

    agent.train(env = environment, epochs = 10000)