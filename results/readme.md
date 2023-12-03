# *Results*

*This folder contains attempts to train these models. However, it requires a considerable amount of computational power, and dealing with Google Colab was pretty annoying.*

*Two games were trained using the provided code, which is why you can see only two environments in `environment.py`. The training took a significant amount of time, and while the results are far from optimal, noticeable progress is evident, and the code functions properly.*

*Atari games do not inherently introduce randomness, so the model shouldn't face any issues mastering the games with extended training and parameter adjustments.*

## *Breakout*

*The initial game trained using the model was Breakout. The game is relatively straightforward, with the ball consistently moving.*

*The improvement by training can be seen in the plot below,it shows the average reward as `Returns` and Games played x 20 as `Epoch x 20`. In the plot it can be seen, that the main improvement happened during the first 1000 - 2000 games, because the epsilon had decayed and exploration was traded with exploitation.*

![text](/results/breakout_results/plots/breakout_plot.png)

**Breakout Gameplay**

* The first GIF showcases gameplay without any training.
* The second GIF displays gameplay after 1000 games.
* The third GIF exhibits gameplay after 3000 games.
* The fourth GIF demonstrates gameplay after 5000 games.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![](/results/breakout_results/breakout_videos/breakout_no_training.gif) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![](/results/breakout_results/breakout_videos/breakout_after_1000_games.gif)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![](/results/breakout_results/breakout_videos/breakout_after_3000_games.gif) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![](/results/breakout_results/breakout_videos/breakout_after_5000_games.gif)

**Explanation**

*As evident from the gameplay, even after 1000 iterations, the agent begins playing better and starts receiving rewards. This occurs because the agent learns that the ball consistently starts from the left side, allowing it to exploit this pattern for rewards. Despite setting the epsilon decay for 2000 games, it continued exploring, resulting in more movement in subsequent gameplay.*

*However, this gameplay is far from optimal, as the agent frequently misses balls and sometimes remains stationary. This indicates that models of this nature require extensive training. Unfortunately, due to limited computational power and time constraints, these results represent the best achievable outcomes.*

## *Pong*

