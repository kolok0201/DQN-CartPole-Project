import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gym
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 確保中文字型能在 Matplotlib 正常顯示 (根據你的作業系統可能需要調整)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.out(x)

class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity, double_dqn=False):
        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory_capacity = memory_capacity
        self.double_dqn = double_dqn # 演算法切換開關

        self.eval_net = Net(n_states, n_actions, n_hidden)
        self.target_net = Net(n_states, n_actions, n_hidden)
        self.target_replace_iter = target_replace_iter
        self.learn_step_counter = 0 
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))
        self.memory_counter = 0

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        x = torch.unsqueeze(torch.tensor(np.array(state), dtype=torch.float), 0)
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action_values = self.eval_net(x)
            action = torch.argmax(action_values).item()
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_state = torch.tensor(b_memory[:, :self.n_states], dtype=torch.float)
        b_action = torch.tensor(b_memory[:, self.n_states:self.n_states+1], dtype=torch.long)
        b_reward = torch.tensor(b_memory[:, self.n_states+1:self.n_states+2], dtype=torch.float)
        b_next_state = torch.tensor(b_memory[:, -self.n_states:], dtype=torch.float)

        # 經驗當時的 Q-value
        q_eval = self.eval_net(b_state).gather(1, b_action)

        # ====== 核心演算法差異區塊 ======
        if self.double_dqn:
            # Double DQN: 用 eval_net 挑選最佳動作，用 target_net 評估該動作的價值
            best_action_next = self.eval_net(b_next_state).argmax(1).unsqueeze(1)
            q_next = self.target_net(b_next_state).gather(1, best_action_next).detach()
            q_target = b_reward + self.gamma * q_next
        else:
            # Standard DQN: 直接用 target_net 找出最大的 Q-value
            q_next = self.target_net(b_next_state).detach()
            q_target = b_reward + self.gamma * q_next.max(1).values.unsqueeze(-1)
        # ================================

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train_agent(algo_name, double_dqn_flag, n_episodes=1500):
    env = gym.make('CartPole-v0').unwrapped
    
    dqn = DQN(n_states=env.observation_space.shape[0],
              n_actions=env.action_space.n,
              n_hidden=128, batch_size=64, lr=0.0005,
              epsilon=1.0, gamma=0.99, target_replace_iter=200, 
              memory_capacity=5000, double_dqn=double_dqn_flag)

    EPSILON_END = 0.05
    EPSILON_DECAY = 0.998
    episode_rewards = []

    print(f"\n🚀 開始訓練: {algo_name}")
    for i_episode in range(n_episodes):
        observation, _ = env.reset()
        rewards = 0

        for t in range(200):
            action = dqn.choose_action(observation)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Reward Shaping
            x, v, theta, omega = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            custom_reward = r1 + r2

            dqn.store_transition(observation, action, custom_reward, next_state)
            rewards += custom_reward

            if dqn.memory_counter > dqn.memory_capacity:
                dqn.learn()

            if done: break
            observation = next_state

        episode_rewards.append(rewards)
        dqn.epsilon = max(EPSILON_END, dqn.epsilon * EPSILON_DECAY)
        
        if (i_episode + 1) % 100 == 0:
            print(f'Episode {i_episode + 1:4d} | Reward: {rewards:6.2f}')
            
    env.close()
    return episode_rewards

# 幫助曲線平滑的輔助函數，讓圖表在報告上更具專業感
def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# === 執行實驗 ===
if __name__ == '__main__':
    episodes = 1500 # 縮短回合數以聚焦於收斂過程
    
    # 分別訓練兩種代理人
    rewards_dqn = train_agent("Standard DQN", double_dqn_flag=False, n_episodes=episodes)
    rewards_ddqn = train_agent("Double DQN", double_dqn_flag=True, n_episodes=episodes)

    # 平滑化數據
    smooth_dqn = moving_average(rewards_dqn)
    smooth_ddqn = moving_average(rewards_ddqn)

    # 繪製對照圖
    plt.figure(figsize=(12, 6))
    
    # 畫出帶有透明度的原始數據背景
    plt.plot(rewards_dqn, alpha=0.2, color='blue', label='DQN (Raw)')
    plt.plot(rewards_ddqn, alpha=0.2, color='orange', label='DDQN (Raw)')
    
    # 畫出平滑後的趨勢線
    plt.plot(smooth_dqn, color='blue', linewidth=2, label='DQN (Moving Average)')
    plt.plot(smooth_ddqn, color='orange', linewidth=2, label='DDQN (Moving Average)')

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward (Custom)', fontsize=12)
    plt.title('DQN vs Double DQN in CartPole Environment', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 儲存高解析度圖片供簡報使用
    plt.savefig('dqn_vs_ddqn_comparison.png', dpi=300)
    plt.show()