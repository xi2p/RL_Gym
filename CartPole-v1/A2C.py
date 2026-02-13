import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

# -------------------- 超参数 --------------------
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR_ACTOR = 1e-3
LR_CRITIC = 5e-3
NUM_EPISODES = 1000
MAX_STEPS = 500
HIDDEN_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Actor 网络 --------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        probs = F.softmax(logits, dim=1)  # 输出动作概率分布
        return probs

# -------------------- Critic 网络 --------------------
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

# -------------------- TD Actor-Critic Agent --------------------
class TDA2CAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic = Critic(state_dim).to(DEVICE)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

    def select_action(self, state):
        """根据当前策略采样动作（不记录梯度）"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            probs = self.actor(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action.item()

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)

        # 计算当前状态的值和下一个状态的值
        value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor)

        # 计算目标值
        with torch.no_grad():
            target = reward + (1 - done) * GAMMA * next_value

        # 更新 Critic
        critic_loss = F.mse_loss(value, torch.tensor([[target]], device=DEVICE))
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # 更新 Actor
        probs = self.actor(state_tensor)
        log_prob = torch.log(probs[0, action])
        advantage = target - value
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        actor_loss = -log_prob * advantage.detach() - 0.01 * entropy
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

# -------------------- 训练 --------------------
def train():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = TDA2CAgent(state_dim, action_dim)

    episode_rewards = []
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        episode_rewards.append(total_reward)

        if (episode+1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode+1}, Average Reward (last 50): {avg_reward:.1f}")

    env.close()

if __name__ == "__main__":
    train()