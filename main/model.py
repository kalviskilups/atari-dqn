import torch
import torch.nn as nn
import os

class AtariNet(nn.Module):
    """
    AtariNet: A PyTorch neural network model for Atari game playing agents.

    Args:
    - nb_actions (int): Number of possible actions in the environment.

    Attributes:
    - relu (torch.nn.ReLU): ReLU activation function.
    - conv1, conv2, conv3 (torch.nn.Conv2d): Convolutional layers.
    - flatten (torch.nn.Flatten): Flatten layer to convert data for fully connected layers.
    - dropout (torch.nn.Dropout): Dropout layer for regularization.
    - action_value1, action_value2, action_value3 (torch.nn.Linear): Fully connected layers for action values.
    - state_value1, state_value2, state_value3 (torch.nn.Linear): Fully connected layers for state values.

    Methods:
    - forward(x): Forward pass through the network.
    - save_the_model(weights_filename): Save the model weights to a file.
    - load_the_model(weights_filename): Load model weights from a file.
    """

    def __init__(self, nb_actions) -> None:
        """
        Initializes the AtariNet class.

        Args:
        - nb_actions (int): Number of possible actions in the environment.
        """

        super(AtariNet, self).__init__()

        # Activation function
        self.relu = nn.ReLU()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size = (8, 8), stride = (4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size = (4, 4), stride = (2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride = (1, 1))

        # Flatten layer
        self.flatten = nn.Flatten()

        # Dropout for regularization
        self.dropout = nn.Dropout(p = 0.2)

        # Fully connected layers for action values
        self.action_value1 = nn.Linear(3136, 1024)
        self.action_value2 = nn.Linear(1024, 1024)
        self.action_value3 = nn.Linear(1024, nb_actions)

        # Fully connected layers for state values
        self.state_value1 = nn.Linear(3136, 1024)
        self.state_value2 = nn.Linear(1024, 1024)
        self.state_value3 = nn.Linear(1024, 1)

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
        - x (torch.Tensor): Input data.

        Returns:
        - output (torch.Tensor): Output tensor from the network.
        """

        x = torch.Tensor(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)

        # Calculating state values
        state_value = self.relu(self.state_value1(x))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value2(state_value))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value3(state_value))

        # Calculating action values
        action_value = self.relu(self.action_value1(x))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.dropout(action_value)
        action_value = self.action_value3(action_value)

        # Final output combining state and action values
        output = state_value + (action_value - action_value.mean())

        return output

    def save_the_model(self, weights_filename = "models/latest.pt"):
        """
        Saves the model weights to a file.

        Args:
        - weights_filename (str): Name of the file to save the weights.
        """

        if not os.path.exists("models"):
            os.mkdir("models")
        torch.save(self.state_dict(), weights_filename)

    def load_the_model(self, weights_filename = "models/latest.pt"):
        """
        Loads model weights from a file.

        Args:
        - weights_filename (str): Name of the file containing the weights to load.
        """

        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f"Success! Loaded {weights_filename}")
        except:
            print(f"No weights available at {weights_filename}")
