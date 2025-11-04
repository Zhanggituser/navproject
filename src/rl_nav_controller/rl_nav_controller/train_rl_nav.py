import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rl_nav_controller.gym_wrapper import RLNavEnv
import matplotlib.pyplot as plt

# PPO超参数
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
LR = 3e-4
BATCH_SIZE = 64
EPOCHS = 10
ROLLOUT_LEN = 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOAgent, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        value = self.critic(x)
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_log_std)
        return mean, std, value

def compute_gae(rewards, masks, values, next_value):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * values[step + 1] * masks[step] - values[step]
        gae = delta + GAMMA * LAMBDA * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def train():
    env = RLNavEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(state_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR)

    all_rewards = []
    timestep = 0
    state = env.reset()

    while timestep < 20000:
        states, actions, log_probs, rewards, masks, values = [], [], [], [], [], []

        # Rollout
        for _ in range(ROLLOUT_LEN):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            mean, std, value = agent(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(1)

            next_state, reward, done, _ = env.step(action.cpu().numpy().flatten())

            states.append(state)
            actions.append(action.cpu().numpy().flatten())
            log_probs.append(log_prob.item())
            rewards.append(reward)
            masks.append(0 if done else 1)
            values.append(value.item())

            state = next_state
            timestep += 1

            if timestep % 100 == 0:
                print(f"[TRAIN] Timestep: {timestep} | Reward: {reward:.2f}")

            if done:
                state = env.reset()

        # Compute returns
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        _, _, next_value = agent(state_tensor)
        returns = compute_gae(rewards, masks, values, next_value.item())
        returns = torch.FloatTensor(returns).to(device)
        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.FloatTensor(actions).to(device)
        old_log_probs_tensor = torch.FloatTensor(log_probs).to(device)
        values_tensor = torch.FloatTensor(values).to(device)

        # PPO update
        for _ in range(EPOCHS):
            mean, std, value = agent(states_tensor)
            dist = torch.distributions.Normal(mean, std)
            entropy = dist.entropy().sum(1).mean()
            new_log_probs = dist.log_prob(actions_tensor).sum(1)

            ratio = (new_log_probs - old_log_probs_tensor).exp()
            advantage = returns - values_tensor
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - value.squeeze(1)).pow(2).mean()

            loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        all_rewards.append(np.sum(rewards))

    # 保存最终模型
    torch.save(agent.state_dict(), "final_model.pth")
    print("Training finished. Final model saved as final_model.pth")

    plt.plot(all_rewards)
    plt.xlabel("Rollouts")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Curve")
    plt.show()

if __name__ == "__main__":
    train()
