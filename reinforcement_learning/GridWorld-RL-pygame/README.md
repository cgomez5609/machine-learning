<h1>Grid World with Pygame - Reinforcement Learning</h1>

This is an implementation of of a Reinforcement learning agent in a grid world. This is based
on the <b>grid world</b> found in AI: A Modern Approach. The idea is for the agent to develop the best
policy from any point in the grid to the goal. 

<h2>Installation</h2>

cd to the directory that contains the requirements.txt, activate a virtual environment and run the following in your terminal or command prompt.

`pip install -r requirements.txt`

Note: I used Python 3.8 and PyQt for this project

<h2>Usage</h2>

Simply run the main method in the main.py file and follow the prompts found in the caption of the pygame window. 
The process is as follows:

1. Pick a node as a goal
2. Pick as many nodes as pits
3. Press Enter to confirm the pits
4. Pick nodes to be obstacles
5. Press Space to confirm
6. Press Space to updates policies
7. The caption will tell you in how many iterations the agent found the best policy
8. Click on any open node to display the best policy (path to goal)

Note: current reward set up all nodes except for the goal, pits, and obstacles are set to -0.04. The goal is set to 1.0 and pits to -1.0. 
The obstales are never used but set to -100.0 to distinguish them. Gamma is set to 0.9. The number of rows is 6 and columns is 8. All these variables
can be adjusted in from constants.py.

Final Note: This implementation was aided by the books: AI A Modern Approach and Deep Rainforcement Learning Hands-On


