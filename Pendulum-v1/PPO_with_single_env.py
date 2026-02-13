import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# ==================== 超参数 ====================
ENV_NAME = "Pendulum-v1"
N_ENVS = 1
T = 2048
EPOCHS = 10
MINIBATCH_SIZE = 64
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
LR = 3e-4
MAX_ITER = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== Actor（高斯策略）====================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def get_dist(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def get_log_prob(self, dist, action):
        return dist.log_prob(action).sum(dim=-1)

# ==================== Critic ====================
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)

# ==================== PPO Agent ====================
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic = Critic(state_dim).to(DEVICE)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=LR
        )

    def get_action_and_value(self, state, action=None):
        dist = self.actor.get_dist(state)
        if action is None:
            action = dist.sample()
        log_prob = self.actor.get_log_prob(dist, action)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(state)
        return action, log_prob, entropy, value

    def evaluate_actions(self, state, action):
        dist = self.actor.get_dist(state)
        log_prob = self.actor.get_log_prob(dist, action)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(state)
        return log_prob, entropy, value

# ==================== 纯Tensor缓冲区 ====================
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def push(self, state, action, reward, done, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value.detach())        # Tensor [1,1] 或 [1]
        self.log_probs.append(log_prob.detach())  # Tensor [1]

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()

    def to_tensor(self, device):
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.FloatTensor(np.array(self.actions)).to(device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(device)
        dones = torch.FloatTensor(np.array(self.dones)).to(device)
        values = torch.cat(self.values, dim=0).flatten()      # [T]
        log_probs = torch.cat(self.log_probs, dim=0).flatten() # [T]
        return states, actions, rewards, dones, values, log_probs

# ==================== 训练 ====================
def train():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(state_dim, action_dim)
    buffer = RolloutBuffer()

    iteration_rewards = []

    for iteration in range(1, MAX_ITER + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_count = 0
        buffer.clear()

        for step in range(T):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(state_tensor)
                action = action.cpu().numpy().flatten()
                log_prob = log_prob.cpu()          # 保持Tensor，转移到cpu以便存储？更好的做法：直接存储Tensor（已在GPU），稍后to_tensor时统一处理
                value = value.cpu()               # 为了与state/action（numpy）统一存储，这里先转为cpu Tensor
                # 注意：为了纯Tensor，我们也可以让state/action也保持Tensor，但为了简单，仍用numpy存储state/action
            # 执行环境
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, done, value, log_prob)

            state = next_state
            episode_reward += reward
            if done:
                state, _ = env.reset()
                episode_count += 1
                iteration_rewards.append(episode_reward)
                episode_reward = 0

        # ---------- 全部转换为Tensor（已在device）----------
        states, actions, rewards, dones, values, old_log_probs = buffer.to_tensor(DEVICE)

        # 计算最后一个状态的价值（bootstrapping）
        with torch.no_grad():
            last_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            last_value = agent.critic(last_state_tensor).flatten()  # [1]

        # 扩展values和dones以包含最后一步
        values_ext = torch.cat([values, last_value])      # [T+1]
        dones_ext = torch.cat([dones, torch.tensor([0.0], device=DEVICE)])  # [T+1]

        # ---------- GAE (纯Tensor循环) ----------
        advantages = torch.zeros(T, device=DEVICE)
        gae = 0
        for t in reversed(range(T)):
            delta = rewards[t] + GAMMA * values_ext[t+1] * (1 - dones_ext[t]) - values_ext[t]
            gae = delta + GAMMA * LAMBDA * (1 - dones_ext[t]) * gae
            advantages[t] = gae
        returns = advantages + values  # [T]

        # ---------- 优化阶段 ----------
        total_loss = 0
        for epoch in range(EPOCHS):
            # 随机打乱
            indices = torch.randperm(T, device=DEVICE)
            for start in range(0, T, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                idx = indices[start:end]

                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_advantages = advantages[idx].unsqueeze(1)   # [batch,1]
                mb_returns = returns[idx].unsqueeze(1)        # [batch,1]
                mb_old_log_probs = old_log_probs[idx].unsqueeze(1)  # [batch,1]

                # 新策略的对数概率、熵、价值
                new_log_probs, entropy, values_pred = agent.evaluate_actions(mb_states, mb_actions)

                ratio = torch.exp(new_log_probs.unsqueeze(1) - mb_old_log_probs)  # [batch,1]

                # clipped surrogate
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(values_pred, mb_returns)

                entropy_loss = -entropy.mean()

                loss = actor_loss + VF_COEF * critic_loss + ENT_COEF * entropy_loss

                agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
                agent.optimizer.step()

                total_loss += loss.item()

        # 日志
        avg_ep_reward = np.mean(iteration_rewards[-max(1, episode_count):]) if episode_count > 0 else 0
        if iteration % 20 == 0:
            print(f"Iter {iteration:3d} | Steps: {iteration * T * N_ENVS:6d} | "
                  f"AvgEpRet: {avg_ep_reward:6.2f} | "
                  f"Loss: {total_loss / (EPOCHS * (T // MINIBATCH_SIZE)):.3f}")

    env.close()
    return iteration_rewards

if __name__ == "__main__":
    rewards = train()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("PPO on Pendulum-v1 (Pure Torch)")
    plt.grid()
    plt.show()