import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time


style.use("ggplot")

ENV_SIZE = 15
NUM_EPISODES = 1000
NUM_STEPS = 50
MOVE_PENALTY = 1
OBSTACLE_PENALTY = 300
GOAL_REWARD = 100
epsilon = 0.9
EPS_DECAY = 0.9998 #0.9998
SHOW_EVERY = 40

start_q_table = None

LEARNING_RATE = 0.05
DISCOUNT = 0.95

JOINT_N = 1
GOAL_N = 2
OBSTACLE_N = 3
MIDPOINT_N = 4

d ={ 1: (255,175,0),
     2 : (0, 255, 0),
     3 : (0, 0, 255),
     4 : (20, 255, 20) 
}

class Blob:
    def __init__(self):
        self.x = np.random.randint(0, ENV_SIZE)
        self.y = np.random.randint(0, ENV_SIZE)
    
    def __str__(self):
        return f"{self.x}, {self.y}"
    
    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)
    
    def action(self, choice):
        #if choice == 0:
            #self.move(x=0, y=-1)
        #elif choice == 1:
            #self.move(x=+1, y=-1)
        if choice == 0:
            self.move(x=-1, y=0)
        elif choice == 1:
            self.move(x=1, y=1)
        elif choice == 2:
            self.move(x=0, y=1)
        elif choice == 3:
            self.move(x=-1, y=1)
        elif choice == 4:
            self.move(x=1, y=0)
        #elif choice == 7:
            #self.move(x=-1, y=-1)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > ENV_SIZE-1:
            self.x = ENV_SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > ENV_SIZE-1:
            self.y = ENV_SIZE-1
print("initializing q_table")
if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for i in range(-ENV_SIZE+1, ENV_SIZE):
        for ii in range(-ENV_SIZE+1, ENV_SIZE):
            for iii in range(-ENV_SIZE+1, ENV_SIZE):
                    for iiii in range(-ENV_SIZE+1, ENV_SIZE):
                        q_table[((i, ii), (iii, iiii))] = [np.random.uniform(-5, 0) for i in range(5)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)
print("q_table ready")
episode_rewards = []
count = 0
for episode in range(NUM_EPISODES):
    print(f"episode: {episode}")
    player = Blob()
    player.y = int(ENV_SIZE/2) - 3
    player.x = ENV_SIZE - 1
    goal = Blob()
    goal.y = int(ENV_SIZE/2) + 3
    goal.x = ENV_SIZE - 1
    obstacle = Blob()
    obstacle.y = int(ENV_SIZE/2)
    obstacle.x = ENV_SIZE - 1
    mid_point = Blob()
    mid_point.y = int(ENV_SIZE/2)
    mid_point.x = ENV_SIZE - 3
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
    episode_reward = 0
    for i in range(NUM_STEPS):
        obs = (player-goal, player-obstacle)
        #print(obs)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 5)
        # Take the action!
        #distance_pre_movement = (((goal.x - player.x)**2 + (goal.y - player.y)**2)**(0.5))
        player.action(action)
        if player.x == obstacle.x and player.y == obstacle.y:
            reward = -OBSTACLE_PENALTY
        elif player.x == goal.x and player.y == goal.y:
            reward = GOAL_REWARD
        elif player.x == mid_point.x and player.y == mid_point.y:
            reward = int(GOAL_REWARD/2) - i
        else:
            reward = -(MOVE_PENALTY * i)
        new_obs = (player-goal, player-obstacle)  # new observation
        max_future_q = np.max(q_table[new_obs])  # max Q value for this new obs
        current_q = q_table[obs][action]

        if reward == GOAL_REWARD:
            new_q = GOAL_REWARD-i
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        if show:
            env = np.zeros((ENV_SIZE, ENV_SIZE, 3), dtype=np.uint8)     # starts an rbg of our size
            env[goal.x][goal.y] = d[GOAL_N]                             # sets the food location tile to green color
            env[player.x][player.y] = d[JOINT_N]                        # sets the player tile to blue
            env[obstacle.x][obstacle.y] = d[OBSTACLE_N]                 # sets the enemy location to red
            env[mid_point.x][mid_point.y] = d[MIDPOINT_N]
            img = Image.fromarray(env, 'RGB')                           # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((800, 800))                              # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))                          # show it!
        if reward == GOAL_REWARD: #or reward == -OBSTACLE_PENALTY:        # crummy code to hang at the end if we reach abrupt end for good reasons or not.
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        episode_reward += reward
        if reward == GOAL_REWARD:
            count += 1
            print(f"!!!goal {count} reached!!! in {i} steps!!!")
        if reward == GOAL_REWARD: #or reward == -OBSTACLE_PENALTY:
            break
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)