import pygame
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque

pygame.init()
pygame.font.init()  # Initialize the font module

# Colors
yellow = pygame.Color(255, 255, 0)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)

# Window Displays
winWidth = 280
winHeight = 280
winDisplay = pygame.display.set_mode((winWidth, winHeight))
pygame.display.set_caption("COMP 469: Venom - Snake Game")

frameSpeed = pygame.time.Clock()

# Snake Initial Size & Location
snakePos = [150, 100]
snakeSpeed = 25
snakeSize = [[150, 100], [140, 100]]
direction = "RIGHT"
score = 0

# Food Initial Random Location
foodPosX = random.randrange(1, int(winWidth / 10)) * 10
foodPosY = random.randrange(1, int(winHeight / 10)) * 10

foodPresent = True

# List to store scores
score_list = []
game_count = 0


memory = deque(maxlen=10000)

# Deep Q-Learning Model
class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearQNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QTrainer:
    def __init__(self, model, targetModel, lr, gamma):
        self.model = model
        self.targetModel = targetModel
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, nextState, done):
        state = torch.tensor(state, dtype=torch.float)
        nextState = torch.tensor(nextState, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            nextState = torch.unsqueeze(nextState, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = torch.unsqueeze(done, 0)

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.targetModel(nextState[idx]))

            target[idx][action[idx].item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

def plot_scores(scores):
    plt.plot(range(len(scores)), scores)
    plt.xlabel('Games')
    plt.ylabel('Scores')
    plt.title('VENOM SCORE TRACKER')
    plt.show()

def scoreCount():
    scoreFont = pygame.font.SysFont("arial", 25)
    scoreDisplay = scoreFont.render("Score : " + str(score), True, white)
    winDisplay.blit(scoreDisplay, [0, 0])

# Ends Game + Displays Final Score
def gameOver():
    global score_list, game_count, score

    # Track the score
    score_list.append(score)

    # Reset score and increment game count
    game_count += 1
    score = 0

    # Reset game state
    reset_game()

def reset_game():
    global snakePos, snakeSize, direction, foodPosX, foodPosY, foodPresent
    snakePos = [150, 100]
    snakeSize = [[150, 100], [140, 100]]
    direction = "RIGHT"
    foodPosX = random.randrange(1, int(winWidth / 10)) * 10
    foodPosY = random.randrange(1, int(winHeight / 10)) * 10
    foodPresent = True

def get_state():
    head_x, head_y = snakePos
    food_x, food_y = foodPosX, foodPosY
    state = [
        head_x, head_y,       
        food_x, food_y,        
        direction == "UP",  
        direction == "DOWN",
        direction == "LEFT",
        direction == "RIGHT",
        head_x - food_x,    
        head_y - food_y,
        len(snakeSize)          
    ]
    return np.array(state, dtype=np.float32)

# Hyperparameters
input_size = 11  
hidden_size = 256
output_size = 4  
learning_rate = 0.002
gamma = 0.9
epsilon = 1.0
decay = 0.995
epsilon_min = 0.01
batch_size = 64

# Initialize Model
model = LinearQNet(input_size, hidden_size, output_size)
targetModel = LinearQNet(input_size, hidden_size, output_size)
targetModel.load_state_dict(model.state_dict())
trainer = QTrainer(model, targetModel, learning_rate, gamma)


target_update_freq = 5
running = True
while running:
    state = get_state()

    if random.random() < epsilon:
        action = random.randint(0, 3)
    else:
        state_tensor = torch.tensor(state, dtype=torch.float)
        action_values = model(state_tensor)
        action = torch.argmax(action_values).item()

    if action == 0:
        direction = "UP"
    elif action == 1:
        direction = "DOWN"
    elif action == 2:
        direction = "LEFT"
    elif action == 3:
        direction = "RIGHT"

    # Changing Position based on Direction
    if direction == "UP":
        snakePos[1] -= 10
    if direction == "DOWN":
        snakePos[1] += 10
    if direction == "RIGHT":
        snakePos[0] += 10
    if direction == "LEFT":
        snakePos[0] -= 10

    # Increases Count if Snake Eats Food
    snakeSize.insert(0, list(snakePos))
    if snakePos[0] == foodPosX and snakePos[1] == foodPosY:
        score += 1
        reward = 10
        foodPresent = False
    else:
        snakeSize.pop()
        reward = -0.1 

    if not foodPresent:
        foodPosX = random.randrange(1, int(winWidth / 10)) * 10
        foodPosY = random.randrange(1, int(winHeight / 10)) * 10
    foodPresent = True

    # Generates New Food in Random Location
    done = False
    if snakePos[0] < 0 or snakePos[0] > winWidth - 10:
        reward = -10
        done = True
    if snakePos[1] < 0 or snakePos[1] > winHeight - 10:
        reward = -10
        done = True

    # Ends Game if Snake Collides with Body
    for snakeBody in snakeSize[1:]:
        if snakePos[0] == snakeBody[0] and snakePos[1] == snakeBody[1]:
            reward = -10
            done = True

    if done:
        gameOver()
        continue

    # Setting Color & Position for Food & Snake
    winDisplay.fill(black)
    for pos in snakeSize:
        pygame.draw.rect(winDisplay, yellow, pygame.Rect(pos[0], pos[1], 10, 10))
    pygame.draw.rect(winDisplay, red, pygame.Rect(foodPosX, foodPosY, 10, 10))

    # Update the display
    scoreCount()
    pygame.display.update()
    frameSpeed.tick(snakeSpeed)

    # Updating Memory
    nextState = get_state()
    memory.append((state, action, reward, nextState, done))

    
    if len(memory) > batch_size:
        mini_batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        trainer.train_step(states, actions, rewards, next_states, dones)

    # Update Target Model
    if game_count % target_update_freq == 0:
        targetModel.load_state_dict(model.state_dict())

    # Update Epsilon
    if epsilon > epsilon_min:
        epsilon *= decay

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Plots Scores when Game Quit
plot_scores(score_list)
pygame.quit()
