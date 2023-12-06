import matplotlib.pyplot as plt
import os

class LivePlot:
    """
    LivePlot: A class for generating live updates of plots for training statistics.

    Attributes:
    - fig (matplotlib.figure.Figure): Matplotlib figure object for plotting.
    - ax (matplotlib.axes.Axes): Matplotlib axes object for data visualization.
    - data (list): List to store average returns over epochs.
    - eps_data (list): List to store epsilon checkpoints over epochs.
    - epochs (int): Number of epochs.

    Methods:
    - __init__(): Initializes the LivePlot class and sets up the figure and axes.
    - update_plot(stats): Updates the plot with the provided statistics.
    """

    def __init__(self) -> None:
        """
        Initializes the LivePlot class.
        """

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Epoch x 20")
        self.ax.set_ylabel("Returns")
        self.ax.set_title("Returns Over Epochs")

        self.data = None
        self.eps_data = None

        self.epochs = 0

    def update_plot(self, stats):
        """
        Updates the plot with the provided statistics.

        Args:
        - stats (dict): Dictionary containing training statistics.
          Expected keys (taken from agent.py): "AvgReturns" for average returns data,
          "EpsilonCheckpoints" for epsilon checkpoints data.
        """

        self.data = stats["AvgReturns"]
        self.eps_data = stats["EpsilonCheckpoints"]

        self.epochs = len(self.data)

        self.ax.clear()
        self.ax.set_xlim(0, self.epochs)
        self.ax.plot(self.data, "b-", label = "Returns")
        self.ax.plot(self.eps_data, "r-", label = "Epsilon")
        self.ax.legend(loc = "upper left")

        if not os.path.exists("plots"):
            os.makedirs("plots")

        self.fig.savefig(f"plots/plot_{self.epochs}.png")
