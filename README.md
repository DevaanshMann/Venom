The Venom Snake Game was developed as a part of my COMP 469 (Introduction to Artificial Intelligence class at California State University, Northridge), in association with my teammates Abigail Rosas, Vanessa Juarez.<br>

The Venom Snake Game is a Python-based implementation of the classic snake game enhanced with artificial intelligence using Deep Q-Learning.<br>
The game uses the `pygame` library for rendering the game graphics and managing user input, while `PyTorch` is used to implement the neural network and training logic for the AI.

**Game Description**:<br>
**Game Objective**: The goal of the game is for the snake to eat food that appears randomly on the game window, which causes the snake to grow longer.<br>
The AI aims to maximize the score by eating as much food as possible without colliding with the walls or its own body.<br>
**Game Window**: The game is displayed in a 280x280 pixel window.<br>
The snake, represented by a yellow square, moves in increments of 10 pixels. The food is represented by a red square, randomly placed on the grid.

**Snake Movement**:<br>
The snake starts with a length of two segments and moves in one of four directions: UP, DOWN, LEFT, or RIGHT.<br>
Each time the snake eats the food, its body length increases, and a new food item is generated at a random location.

**Game Over Conditions**:<br>
The game ends if the snake's head collides with the boundaries of the window or with its own body.<br>
Upon game over, the score is recorded, and the game state is reset for a new round.

**AI Description**:<br>
**Deep Q-Learning**:<br>
The AI controls the snake using a Deep Q-Network (DQN), which learns by interacting with the game environment.<br>
It observes the state of the game (snake position, food position, and direction) and takes actions (move UP, DOWN, LEFT, RIGHT) to maximize the long-term reward.

**Model Architecture**:<br>
The neural network consists of three fully connected layers:<br>
- Input layer: Takes 11 features representing the game state.<br>
- Two hidden layers with ReLU activations for non-linearity.<br>
- Output layer: Provides the Q-values for the four possible actions (UP, DOWN, LEFT, RIGHT).

**Training Process**:<br>
The AI is trained using experience replay.<br>
Each time step, the AI stores the current state, action, reward, next state, and whether the game ended in a replay memory buffer.<br>
The Q-Network is updated by sampling mini-batches from this buffer and training the model to predict the optimal future rewards using the Bellman equation.<br>
A separate target network is used for more stable training, which is updated periodically to match the weights of the current model.

**Exploration vs. Exploitation**:<br>
The AI starts by exploring random actions to learn about the environment.<br>
Over time, the exploration rate decays, and the AI increasingly exploits the learned Q-values to make more informed decisions.

**Scoring**:<br>
- The AI receives a positive reward when the snake eats food (+10 points).<br>
- It gets a negative reward for hitting a wall or colliding with itself (-10 points) or for not eating food (-0.1 per move).

**Score Plotting**:<br>
After the game session ends, the scores from each round are plotted using `matplotlib`, showing the AI's performance over time.

**Summary**:<br>
This implementation combines traditional game mechanics with modern AI techniques to create an autonomous snake that learns how to play the game better with each round.<br>
The Deep Q-Learning AI gradually improves by learning from its mistakes and optimizing its gameplay strategies.
