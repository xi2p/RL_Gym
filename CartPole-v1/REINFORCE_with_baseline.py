import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# -------------------- 超参数 --------------------
ENV_NAME = "CartPole-v1"
NUM_EPISODES = 500
MAX_STEPS = 500
GAMMA = 0.99
LR_ACTOR = 1e-3          # Actor 学习率
LR_CRITIC = 1e-2         # Critic 学习率（通常可设更大）
HIDDEN_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Actor 网络 --------------------
class Actor(nn.Module):
    """策略网络：输出动作概率分布"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)           # 未归一化的对数概率
        return logits

# -------------------- Critic 网络 --------------------
class Critic(nn.Module):
    """价值网络：输出状态价值 V(s)"""
    def __init__(self, state_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# -------------------- A2C 智能体（独立网络）--------------------
class A2CAgent:
    def __init__(self, state_dim, action_dim):
        # 初始化两个独立的网络
        self.actor = Actor(state_dim, action_dim, HIDDEN_DIM).to(DEVICE)
        self.critic = Critic(state_dim, HIDDEN_DIM).to(DEVICE)

        # 两个网络使用不同的优化器（也可共用一个优化器，但参数列表不同）
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

    def select_action(self, state):
        """根据当前策略采样动作，返回动作、对数概率、状态价值"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        # Actor 前向
        logits = self.actor(state_tensor)
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Critic 前向
        value = self.critic(state_tensor)

        return action.item(), log_prob, value

    def update(self, log_probs, values, rewards):
        """回合结束后，计算折扣回报和优势，分别更新 Actor 和 Critic"""
        # 1. 计算折扣回报 G_t (蒙特卡洛)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE).view(-1, 1)

        # 2. 将 values 堆叠为张量
        values = torch.cat(values, dim=0)                # [ep_len, 1]

        # 3. 计算优势 A_t = G_t - V(s_t) （Critic 梯度不流入 Actor）
        advantages = returns - values.detach()

        # 4. 准备 log_probs 张量
        log_probs = torch.cat(log_probs, dim=0).view(-1, 1)

        # ---------- 更新 Actor ----------
        actor_loss = -(log_probs * advantages).mean()    # 策略梯度
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # ---------- 更新 Critic ----------
        critic_loss = F.mse_loss(values, returns)        # 价值估计误差
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

# -------------------- 训练主循环 --------------------
def train():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = A2CAgent(state_dim, action_dim)

    episode_rewards = []
    moving_avg = []

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0

        log_probs = []
        values = []
        rewards = []

        for step in range(MAX_STEPS):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state
            episode_reward += reward

            if done:
                break

        # 回合结束，更新网络
        agent.update(log_probs, values, rewards)

        episode_rewards.append(episode_reward)

        if len(episode_rewards) >= 10:
            avg = np.mean(episode_rewards[-10:])
            moving_avg.append(avg)
        else:
            moving_avg.append(episode_reward)

        if (episode+1) % 50 == 0:
            print(f"Episode {episode+1}, Reward: {episode_reward:.1f}")

    env.close()

    # 绘图
    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(moving_avg, label='10-episode Moving Avg', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('REINFORCE on CartPole')
    plt.legend()
    plt.grid(True)
    plt.savefig('REINFORCE_cartpole.png')
    plt.show()

if __name__ == "__main__":
    train()