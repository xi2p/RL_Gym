import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt
import tqdm

# ==================== 超参数 ====================
ENV_NAME = "Pendulum-v1"
N_ENVS = 1
T = 2048
EPOCHS = 10
MINIBATCH_SIZE = 64
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.1
VF_COEF = 0.5
ENT_COEF = 0.0001
LR = 1e-4
MAX_ITER = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== Actor（高斯策略）====================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        mean = self.net(state)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def get_dist(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def get_log_prob(self, dist, action):
        return dist.log_prob(action).sum(dim=-1).unsqueeze(1)

# ==================== Critic ====================
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
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

    def select_action(self, state):
        dist = self.actor.get_dist(state)
        action = dist.sample()
        log_prob = self.actor.get_log_prob(dist, action)
        value = self.critic(state)
        return action, log_prob, value

    def evaluate_actions(self, states, actions):
        dists = self.actor.get_dist(states)
        log_prob = self.actor.get_log_prob(dists, actions)
        entropy = dists.entropy().sum(dim=-1)
        value = self.critic(states)
        return log_prob, value, entropy

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
        self.values.append(value.detach())        # Tensor [1,1]
        self.log_probs.append(log_prob.detach())  # Tensor [1,1]

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()

    def to_tensor(self, device):
        states = torch.cat(self.states, dim=0).to(device)
        actions = torch.cat(self.actions, dim=0).to(device)
        rewards = torch.FloatTensor(self.rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(self.dones).unsqueeze(1).to(device)
        values = torch.cat(self.values, dim=0).to(device)
        log_probs = torch.cat(self.log_probs, dim=0).to(device)
        return states, actions, rewards, dones, values, log_probs

# ==================== 训练 ====================
def train():
    env = gym.make(ENV_NAME, max_episode_steps=2048)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(state_dim, action_dim)

    # Load trained parameters
    agent.actor.load_state_dict(torch.load("ppo_actor.pth"))
    agent.critic.load_state_dict(torch.load("ppo_critic.pth"))
    agent.optimizer.load_state_dict(torch.load("ppo_optim.pth"))

    buffer = RolloutBuffer()

    iteration_rewards = []

    for iteration in tqdm.trange(1, MAX_ITER + 1, desc="Training PPO"):
        state, _ = env.reset(options={"low": -0.03, "high": 0.03})
        episode_reward = 0
        buffer.clear()

        # ---------- 与环境交互 收集轨迹 ----------
        for step in range(T):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

            # select an action
            with torch.no_grad():
                action, log_prob, value = agent.select_action(state_tensor)

            # act
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            done = terminated or truncated

            # state_tensor: [1, state_dim],
            # action:       [1, action_dim],
            # reward:       scalar,
            # done:         bool,
            # value:        [1,1],
            # log_prob:     [1,1]

            buffer.push(state_tensor, action, reward, done, value, log_prob)

            state = next_state
            episode_reward += reward
            if done:
                state, _ = env.reset(options={"low": -0.03, "high": 0.03})
                iteration_rewards.append(episode_reward)

        # ---------- 全部转换为Tensor----------
        states, actions, rewards, dones, values, old_log_probs = buffer.to_tensor(DEVICE)
        # All the above are [T, ...] shape
        # Specifically:
        # states:        [T, state_dim]
        # actions:       [T, action_dim]
        # rewards:       [T, 1]
        # dones:         [T, 1]
        # values:        [T, 1]
        # old_log_probs: [T, 1]

        # 计算最后一个状态的价值
        with torch.no_grad():
            last_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)    # [1, state_dim]
            last_value = agent.critic(last_state_tensor)    # [1, 1]

        # 扩展values和dones以包含最后一步
        values_ext = torch.cat([values, last_value], dim=0)   # [T+1, 1]
        dones_ext = torch.cat([dones, torch.tensor([[0.0]], device=DEVICE)], dim=0)  # [T+1, 1]

        # ---------- GAE ----------
        advantages = torch.zeros((T,1), device=DEVICE)
        gae = 0.0
        for t in reversed(range(len(values))):
            delta = rewards[t] + GAMMA * values_ext[t+1] * (1 - dones_ext[t]) - values_ext[t]
            gae = delta + GAMMA * LAMBDA * (1 - dones_ext[t]) * gae
            advantages[t] = gae
        returns = advantages + values  # [T, 1]

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
                mb_advantages = advantages[idx]   # [batch,1]
                mb_returns = returns[idx]        # [batch,1]
                mb_old_log_probs = old_log_probs[idx]  # [batch,1]

                # 用现在的策略重新评估动作，主要是看log_prob的变化。
                new_log_probs, values_pred, entropy = agent.evaluate_actions(mb_states, mb_actions)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)  # [batch,1]

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
        if iteration % 5 == 0:
            print(f"\nIter {iteration:3d} | Steps: {iteration * T * N_ENVS:6d} | "
                  f"Loss: {total_loss / (EPOCHS * (T // MINIBATCH_SIZE)):.3f} | "
                  f"Ep Reward: {iteration_rewards[-1]:.2f}")
            # 保存参数
            torch.save(agent.actor.state_dict(), "ppo_actor.pth")
            torch.save(agent.critic.state_dict(), "ppo_critic.pth")
            torch.save(agent.optimizer.state_dict(), "ppo_optim.pth")
    env.close()
    return iteration_rewards

if __name__ == "__main__":
    rewards = train()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("PPO on Pendulum-v1 (Pure Torch)")
    plt.grid()
    plt.savefig('ppo_pendulum.png')
    plt.show()