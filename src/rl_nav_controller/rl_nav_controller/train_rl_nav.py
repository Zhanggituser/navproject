#!/usr/bin/env python3
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import rclpy
from rl_nav_controller.gym_wrapper import RLNavEnv
import matplotlib.pyplot as plt


# PPO 超参数（稳定训练配置）
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.001
VALUE_COEF = 0.5
LR = 1e-4            # 降低学习率
ROLLOUT_LEN = 1024   # 减小 rollout 长度
EPOCHS = 4
MAX_TIMESTEPS = 100000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOAgent, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        value = self.critic(x)
        mean = self.actor_mean(x)
        # 防止 std 过大或过小
        std = torch.exp(torch.clamp(self.actor_log_std, -20, 2))
        return mean, std, value


def compute_gae(rewards, masks, values, next_value):
    """计算 GAE，带数值保护"""
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        reward = rewards[step] if np.isfinite(rewards[step]) else 0.0
        val = values[step] if np.isfinite(values[step]) else 0.0
        next_val = values[step + 1] if np.isfinite(values[step + 1]) else 0.0
        
        delta = reward + GAMMA * next_val * masks[step] - val
        gae = delta + GAMMA * LAMBDA * masks[step] * gae
        returns.insert(0, gae + val)
    return returns


def main():
    rclpy.init()
    env = RLNavEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(state_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    all_rewards = []
    timestep = 0
    state, _ = env.reset()

    try:
        while timestep < MAX_TIMESTEPS:
            states, actions, log_probs, rewards, masks, values = [], [], [], [], [], []

            # Rollout
            for _ in range(ROLLOUT_LEN):
                if timestep >= MAX_TIMESTEPS:
                    break

                # 确保 state 有效
                if not np.all(np.isfinite(state)):
                    state = np.zeros_like(state)

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    mean, std, value = agent(state_tensor)
                    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                        state, _ = env.reset()
                        continue

                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(1)

                next_state, reward, done, info = env.step(action.cpu().numpy().flatten())

                # 裁剪 reward 防止极端值
                reward = np.clip(reward, -20.0, 50.0)

                states.append(state.copy())
                actions.append(action.cpu().numpy().flatten().copy())
                log_probs.append(log_prob.item())
                rewards.append(float(reward))
                masks.append(0.0 if done else 1.0)
                values.append(float(value.item()))

                state = next_state
                timestep += 1

                if done:
                    state, _ = env.reset()

            if len(states) == 0:
                continue

            # 转换为 numpy array 再转 tensor（解决警告 + 提高速度）
            states_np = np.array(states, dtype=np.float32)
            actions_np = np.array(actions, dtype=np.float32)
            log_probs_np = np.array(log_probs, dtype=np.float32)
            values_np = np.array(values, dtype=np.float32)
            returns_np = np.array(compute_gae(rewards, masks, values, 0.0), dtype=np.float32)

            states_tensor = torch.FloatTensor(states_np).to(device)
            actions_tensor = torch.FloatTensor(actions_np).to(device)
            old_log_probs_tensor = torch.FloatTensor(log_probs_np).to(device)
            returns_tensor = torch.FloatTensor(returns_np).to(device)
            values_tensor = torch.FloatTensor(values_np).to(device)  # shape: [N]

            # PPO update
            for _ in range(EPOCHS):
                mean, std, value = agent(states_tensor)  # value: [N, 1]
                
                if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                    print("NaN detected in policy. Skipping update.")
                    break

                dist = torch.distributions.Normal(mean, std)
                entropy = dist.entropy().sum(1).mean()
                new_log_probs = dist.log_prob(actions_tensor).sum(1)

                ratio = (new_log_probs - old_log_probs_tensor).exp()
                # ✅ 修复：values_tensor 是 1D，直接相减
                advantage = returns_tensor - values_tensor  # both [N]
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantage
                actor_loss = -torch.min(surr1, surr2).mean()
                # ✅ value.squeeze() 将 [N,1] → [N]
                critic_loss = (returns_tensor - value.squeeze()).pow(2).mean()

                loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                optimizer.step()

            all_rewards.append(np.sum(rewards))
            if timestep % 1000 == 0:
                avg_reward = np.mean(all_rewards[-10:]) if len(all_rewards) >= 10 else np.mean(all_rewards)
                print(f"[Timestep {timestep}] Avg Reward (last 10): {avg_reward:.2f}")

        # 保存模型
        torch.save(agent.state_dict(), "final_model.pth")
        print("✅ Training finished. Model saved as 'final_model.pth'")

        # 绘制奖励曲线
        plt.figure(figsize=(10, 6))
        plt.plot(all_rewards)
        plt.xlabel("Rollout")
        plt.ylabel("Total Reward")
        plt.title("PPO Training Reward Curve")
        plt.grid(True)
        plt.savefig("training_rewards.png")
        print("📈 Reward curve saved as 'training_rewards.png'")

    finally:
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()