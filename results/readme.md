# Results

This folder contains the data of the attempts to train these models. You can see the tests and plots below and in the data folder. The results are not optimal as the training takes a lot of time and in this case "Breakout" and "Space Invaders" were trained for 5000 games and "Pong" was trained for 3000 games.

Only the three mentioned games were trained using the provided code, which is why you can see only three environments in `environment.py`. The training took a significant amount of time, and while the results are far from optimal, noticeable progress is evident, and the code functions properly.

The improvement by training can be seen in the plot, it shows the average reward as `Returns` and Games played x 20 as `Epoch x 20`.

The .gif files show the gameplay that was achieved by testing after different amount of games:

* The first GIF showcases gameplay without any training (top left).
* The second GIF displays gameplay after 1000 games (top right).
* The third GIF exhibits gameplay after 3000 (2000 for "Pong") games (bottom left).
* The fourth GIF demonstrates gameplay after 5000 (3000 for "Pong") games (bottom right).

<br/>

## Breakout

### Plot of returns over epochs x 20

<br/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![text](/results/data/breakout/breakout_plot.png)

<br/>

### Gameplay

<br/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![](/results/data/breakout/breakout_0.gif) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![](/results/data/breakout/breakout_1000.gif)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![](/results/data/breakout/breakout_3000.gif) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![](/results/data/breakout/breakout_5000.gif)

<br/>

## Pong

### Plot of returns over epochs x 20

<br/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![text](/results/data/pong/pong_plot.png)

<br/>

### Gameplay

<br/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![](/results/data/pong/pong_0.gif) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![](/results/data/pong/pong_1000.gif)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![](/results/data/pong/pong_2000.gif) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![](/results/data/pong/pong_3000.gif)

<br/>

## Space Invaders

### Plot of returns over epochs x 20

<br/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![text](/results/data/spaceinvaders/spaceinvaders_plot.png)

<br/>

### Gameplay

<br/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![](/results/data/spaceinvaders/spaceinvaders_0.gif) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![](/results/data/spaceinvaders/spaceinvaders_1000.gif)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![](/results/data/spaceinvaders/spaceinvaders_3000.gif) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![](/results/data/spaceinvaders/spaceinvaders_5000.gif)

<br/>
