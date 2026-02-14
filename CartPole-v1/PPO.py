import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt
import tqdm

# ==================== 超参数 ====================
ENV_NAME = "CartPole-v1"
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
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        logits = self.net(state)
        return F.softmax(logits, dim=-1)

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

    def select_action(self, state: torch.Tensor):
        with torch.no_grad():
            probs = self.actor(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
        return action, probs[:, action], self.critic(state)

    def evaluate_actions(self, states, actions):
        # states:   [batch, state_dim]
        # actions:  [batch, 1]
        probs = self.actor(states).gather(1, actions)
        values = self.critic(states)
        return probs, values, probs*torch.log(probs)



# ==================== 纯Tensor缓冲区 ====================
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.probs = []

    def push(self, state, action, reward, done, value, prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value.detach())        # Tensor [1,1]
        self.probs.append(prob.detach())  # Tensor [1,1]

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.probs.clear()

    def to_tensor(self, device):
        states = torch.cat(self.states, dim=0).to(device)
        actions = torch.tensor(self.actions, dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(self.rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(self.dones).unsqueeze(1).to(device)
        values = torch.cat(self.values, dim=0).to(device)
        probs = torch.cat(self.probs, dim=0).unsqueeze(1).to(device)
        return states, actions, rewards, dones, values, probs

# ==================== 训练 ====================
def train():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)
    buffer = RolloutBuffer()

    iteration_rewards = []

    for iteration in tqdm.trange(1, MAX_ITER + 1, desc="Training PPO"):
        state, _ = env.reset()
        episode_reward = 0
        episode_count = 0
        buffer.clear()

        # ---------- 与环境交互 收集轨迹 ----------
        for step in range(T):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

            # select an action
            with torch.no_grad():
                action, prob, value = agent.select_action(state_tensor)

            # act
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # state_tensor: [1, state_dim]
            # action:       scalar
            # reward:       scalar
            # done:         bool
            # value:        [1, 1]
            # prob:         [1, ]

            buffer.push(state_tensor, action, reward, done, value, prob)

            state = next_state
            episode_reward += reward
            if done:
                state, _ = env.reset()
                episode_count += 1

        iteration_rewards.append(episode_reward / episode_count)

        # ---------- 全部转换为Tensor----------
        states, actions, rewards, dones, values, old_probs = buffer.to_tensor(DEVICE)
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
                mb_old_probs = old_probs[idx]  # [batch,1]

                # 用现在的策略重新评估动作，主要是看log_prob的变化。
                new_probs, values_pred, entropy = agent.evaluate_actions(mb_states, mb_actions)

                ratio = new_probs / mb_old_probs  # [batch,1]

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

    env.close()
    return iteration_rewards

if __name__ == "__main__":
    rewards = train()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("PPO on CartPole-v1")
    plt.grid()
    plt.savefig('ppo_cartpole.png')
    plt.show()