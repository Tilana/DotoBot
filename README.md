# DotoBot

As I couldn't beat my flat mates in playing Dotowheel, I programmed a bot to do this.



### Dotowheel

[Dotowheel](https://play.google.com/store/apps/details?id=com.nebulabytes.dotowheel&hl=en) is a fun game with simple rules:

Place dots on the wheel. When at least three dots with the same number are next to each other they merge to the next higher number. Get as high as possible!

If you run out of empty space on the wheel you loose.

Joker: Every 5th move you have the possibility to swap two dots.


[![Watch the video](https://github.com/Tilana/DotoBot/blob/master/screenshots/image002.png)](https://youtu.be/gSXJhuXVR-M)



### Setup

Dotobot is based on *Python 2.7* and tensorflow 1.14.

Create a  virtual environment with Python 2.7 and install the requirements:

```
pip install -r requirements.txt
```

Start DotoBot with:

``python script.py ``



As there is no open-source version of the Dotowheel code available a browser window with the online game is loaded. DotoBot is then analyzing the current state of the game based on the dot colors. It places the dots by controlling the mouse. The position within the wheel are hard-coded.



### DotoBot

DotoBot is based on reinforcement learning. 

In reinforcement learning an agent learns to solve a problem in a complex environment. Based on the performed actions it can get a reward or a penalty. While in the beginning the actions are random, by maximizing the reward the agent can develop sophisticated strategies to solve a problem.

In this case, DotoBot has the goal to maximize its score and therefore reach the highest number as possible on the wheel. It gets a reward if dots are merged and is penalized if the game is lost. Besides the valid possibilities for the current move, DotoBot does not know anything about the game.

More specifically, DotoBot's strategy is based on Q-Learning which is implemented with a two layer convolution network.



If you wanna learn more about reinforcement learning, have a look at [this introduction by Pathmind](https://pathmind.com/wiki/deep-reinforcement-learning).



