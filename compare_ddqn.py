import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gym
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 定義神經網路結構 [cite: 125, 126]
class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__() # [cite: 128]
        self.fc1 = nn.Linear(n_states, n_hidden) # [cite: 129]
        self.out = nn.Linear(n_hidden, n_actions) # [cite: 130]
        nn.init.xavier_normal_(self.fc1.weight) # [cite: 131]
        nn.init.xavier_normal_(self.out.weight) # [cite: 132]

    def forward(self, x):
        x = self.fc1(x) # [cite: 134]
        x = F.relu(x) # [cite: 135]
        action_values = self.out(x) # [cite: 136]
        return action_values # [cite: 137]

# 定義 DQN 演算法主體 [cite: 140]
class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity): # [cite: 141]
        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory_capacity = memory_capacity

        self.eval_net = Net(n_states, n_actions, n_hidden) # [cite: 142]
        self.target_net = Net(n_states, n_actions, n_hidden) # [cite: 143]
        self.target_replace_iter = target_replace_iter  # target net 多久 update 一次 [cite: 144]
        self.learn_step_counter = 0  # 現在學多久了 [cite: 145]
        
        # memory buffer, 每一筆經驗是 (state + next state + reward + action) [cite: 146]
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2)) # [cite: 147]
        self.memory_counter = 0  # buffer 中幾筆經驗了 [cite: 148]

        # 其他訓練需要的 [cite: 149]
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr) # [cite: 150]
        self.loss_func = nn.MSELoss() # [cite: 151]

    # 選擇 action，使用 epsilon-greedy 策略 [cite: 153, 154]
    def choose_action(self, state):
        x = torch.unsqueeze(torch.tensor(np.array(state), dtype=torch.float), 0) # [cite: 155]
        if np.random.uniform() < self.epsilon: # [cite: 156]
            action = np.random.randint(0, self.n_actions) # [cite: 157]
        else: # [cite: 158]
            # eval net 預測 Q-value [cite: 159]
            action_values = self.eval_net(x) # [cite: 160]
            # 選 Q-value 最大的 action [cite: 161]
            action = torch.argmax(action_values).item() # [cite: 162]
        return action # [cite: 163]

    # 將經驗存起來 (根據文件框架補齊實作細節) [cite: 193, 194]
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    # 當 buffer 有足夠經驗，我們讓 eval net 學習 [cite: 164, 165]
    def learn(self):
        # 從 buffer 中隨機挑選經驗，將經驗分成 state、action、reward、next state [cite: 166]
        sample_index = np.random.choice(self.memory_capacity, self.batch_size) # [cite: 167]
        b_memory = self.memory[sample_index, :] # [cite: 168]
        b_state = torch.tensor(b_memory[:, :self.n_states], dtype=torch.float) # [cite: 169]
        b_action = torch.tensor(b_memory[:, self.n_states:self.n_states+1], dtype=torch.long) # [cite: 170]
        b_reward = torch.tensor(b_memory[:, self.n_states+1:self.n_states+2], dtype=torch.float) # [cite: 171]
        b_next_state = torch.tensor(b_memory[:, -self.n_states:], dtype=torch.float) # [cite: 172]

        # 計算 eval net 的 Q-value 和 target net 的 loss [cite: 173]
        q_eval = self.eval_net(b_state).gather(1, b_action)  # 經驗當時的 Q-value [cite: 174]
        q_next = self.target_net(b_next_state).detach() # [cite: 175]
        q_target = b_reward + self.gamma * q_next.max(1).values.unsqueeze(-1)  # 目標 Q-value [cite: 176]
        loss = self.loss_func(q_eval, q_target) # [cite: 177]

        # Backpropagation [cite: 178]
        self.optimizer.zero_grad() # [cite: 179]
        loss.backward() # [cite: 180]
        self.optimizer.step() # [cite: 181]

        # Target network 一陣子更新一次 [cite: 182]
        self.learn_step_counter += 1 # [cite: 183]
        if self.learn_step_counter % self.target_replace_iter == 0: # [cite: 184]
            self.target_net.load_state_dict(self.eval_net.state_dict()) # [cite: 185]


# === 主訓練迴圈 ===

# 建立環境 [cite: 36, 37]
env = gym.make('CartPole-v0')
env = env.unwrapped  # 為了在後續取得 env.x_threshold 等內部參數

n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# RL 訓練長度
N_EPISODES = 10000
EPISODE_LENGTH = 200

# Epsilon decay 參數
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.998  # 放慢衰減，給更多時間探索

# 建立 DQN
dqn = DQN(n_states=n_states,
          n_actions=n_actions,
          n_hidden=128,
          batch_size=64,
          lr=0.0005,          # 降低學習率，減少震盪
          epsilon=EPSILON_START,
          gamma=0.99,
          target_replace_iter=200,  # 降低 target net 更新頻率，提高穩定性
          memory_capacity=5000)

# 開始訓練
episode_rewards = []

for i_episode in range(N_EPISODES):
    observation, _ = env.reset()
    rewards = 0

    for t in range(EPISODE_LENGTH):
        action = dqn.choose_action(observation)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 自訂 reward：車子和柱子越靠中間越棒
        x, v, theta, omega = next_state
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2

        dqn.store_transition(observation, action, reward, next_state)
        rewards += reward

        if dqn.memory_counter > dqn.memory_capacity:
            dqn.learn()

        if done:
            break

        observation = next_state

    episode_rewards.append(rewards)

    # Epsilon decay：每回合結束後降低探索率
    dqn.epsilon = max(EPSILON_END, dqn.epsilon * EPSILON_DECAY)

    print('Episode {:3d} | Steps: {:3d} | Reward: {:6.2f} | Epsilon: {:.3f}'.format(
        i_episode + 1, t + 1, rewards, dqn.epsilon))

env.close() # 完成訓練後把環境完整關閉 [cite: 19, 53]

# 畫出獎勵曲線
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label='每回合獎勵')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN CartPole Reward Curve')
plt.legend()
plt.tight_layout()
plt.savefig('rewards.png')
plt.show()