# RL_Gym

`RL_Gym` is a project that implements and benchmarks various reinforcement learning algorithms for training agents in environments from the [Gymnasium](https://gymnasium.farama.org/) library (formerly OpenAI Gym).

---

## Environments & Algorithms

### CartPole-v1

A classic control problem where a pole is attached to a cart moving along a frictionless track. The agent must apply left or right forces to keep the pole upright while preventing the cart from moving off-screen.

- **Action space**: Discrete  
- **Algorithms used**:
  - A2C
  - DQN
  - REINFORCE with baseline

### Pendulum-v1

A continuous control task involving an underactuated pendulum that starts at a random angle. The agent applies torque to swing the pendulum up and stabilize it in the upright (vertical) position.

- **Action space**: Continuous  
- **Algorithms used**:
  - DDPG
  - PPO