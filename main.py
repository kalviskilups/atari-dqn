import os
from breakout import *

import gym
import torch
import numpy as np
from PIL import Image
from model import AtariNet
from agent import Agent

#Dueling DQN

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

environment = DQNBreakout(device = device)

model = AtariNet(nb_actions = 4)

model.to(device)

model.load_the_model()

agent = Agent(model = model, 
              device = device,
              epsilon = 1,
              nb_warmup = 1000,
              nb_actions = 4,
              learning_rate = 0.00025,
              memory_capacity = 500000,
              batch_size = 128)

agent.train(env = environment, epochs = 2000)