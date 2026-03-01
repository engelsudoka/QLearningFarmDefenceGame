# QLearningFarmDefenceGame
ML project that designs and implements a game that utilizes DQN-learning, a model-free reinforcement learning algorithm, that trains an autonomous agent to make optimal decisions based on the game environment and enemy  motion pattern.

GAME OVERVIEW:
--------------
A farm defense game where an AI agent (brown square / custom sprite) learns
to protect crops (green circles) from invading boars (black+red squares) by
shooting bullets.

REINFORCEMENT LEARNING COMPONENTS:
-----------------------------------
1. STATES  (8 continuous features, normalised to [0,1]):
   shooter_x, shooter_y, closest_boar_x, closest_boar_y,
   distance_to_boar, angle_to_boar, num_boars, ai_bullet_count

2. ACTIONS (10 discrete):
   0-7: move in 8 directions   |  8: Momentarily stay stationary |   9: shoot at nearest boar

3. REWARDS:
   +100  hit and eliminate a boar
     -5  bullet goes off-screen (miss)
     -1  every timestep (efficiency pressure)
    -10  boar reaches a vegetable

DQN ARCHITECTURE:
   Input(8) → FC(128,ReLU) → FC(64,ReLU) → Output(9) ~ 9792 trainable parameters

DQN ALGORITHM:
   Experience replay buffer (10 000 transitions)
   Separate target network (synced every 100 steps)
   Epsilon-greedy exploration, Adam optimiser (lr = 0.001)
![FDG](https://github.com/user-attachments/assets/dcff70a5-5cae-4b8e-927b-13d7ccd4debb)
