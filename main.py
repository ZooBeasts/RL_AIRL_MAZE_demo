import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import Network
import RewardEnv
import environment
from utils import ExpertDataset, save_ppo_trajectories
from wrapped_net_func import train_reward_network, train_discriminator
from stable_baselines3.common.callbacks import BaseCallback

Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TOTAL_STEPS = 15000  # PPO timesteps
EXPERT_EPOCHS = 300  # Offline training epochs for RewardNet
BC_EPOCHS = 1200  # Behavioral cloning epochs
IRL_ITERS = 10000  # Adversarial / IRL iterations (define as needed)
initial_bc_coef = 0.05
final_bc_coef = 0.1

data = np.load("expert_set.npz", allow_pickle=True)  # Replace with your file path
expert_obs = expert_states = data["states"]  # List of state sequences
expert_actions = EXPERT_ACTIONS = [np.array(seq, dtype=int) for seq in data["actions"]]  # List of action sequences
expert_r = data["rewards"]

assert len(expert_obs) == len(expert_actions) == len(expert_r), "Expert data lengths do not match!"

max_len = max(len(traj) for traj in expert_states)
padded_states = [
    np.pad(
        traj,
        ((0, max_len - len(traj)), (0, 0)),  # 仅填充时间维度
        mode='constant'
    )
    for traj in expert_states
]

padded_actions = [
    np.pad(
        traj,
        (0, max_len - len(traj)),
        mode='constant'
    )
    for traj in EXPERT_ACTIONS
]

expert_obs_tensor = torch.tensor(np.stack(padded_states), dtype=torch.float32, device=Device)
expert_actions_tensor = torch.tensor(np.stack(padded_actions), dtype=torch.long, device=Device)

# 将动作转换为 one-hot 编码，增加 action_dim 维度
action_onehot = F.one_hot(expert_actions_tensor, num_classes=4).float()  # [batch_size, max_len, action_dim]

padded_r = [
    np.pad(traj, (0, max_len - len(traj)),
           mode='constant'
           ) for traj in expert_r]

expert_r_tensor = torch.tensor(np.stack(padded_r), dtype=torch.float32, device=Device).unsqueeze(-1)
# 在预热阶段归一化专家奖励
expert_r_tensor = (expert_r_tensor - expert_r_tensor.min()) / (expert_r_tensor.max() - expert_r_tensor.min())

padded_obs = [
    np.pad(
        traj,
        ((0, max_len - len(traj)), (0, 0)),
        mode='constant',
        constant_values=0
    ).astype(np.float32)
    for traj in expert_obs
]
padded_actions = [
    np.pad(
        act,
        (0, max_len - len(act)),
        mode='constant',
        constant_values=0
    ).astype(np.int64)
    for act in expert_actions
]

# 转换为numpy数组
expert_obs = np.stack(padded_obs)  # [num_traj, max_len, 8]
expert_actions = np.stack(padded_actions)  # [num_traj, max_len]

# Initialize rewards and discriminator networks
r_model = Network.RewardNetwork(8, 4, 256).to(Device)
optimizer = torch.optim.Adam(r_model.parameters(), lr=1e-4)

discriminator = Network.Discriminator(state_dim=8, action_dim=4, hidden_dim=512).to(Device)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

# 训练 RewardNetwork
for _ in range(EXPERT_EPOCHS):
    pred = r_model(expert_obs_tensor, action_onehot)  # 传入 one-hot 编码后的动作
    loss_r = F.mse_loss(pred, expert_r_tensor)
    optimizer.zero_grad()
    loss_r.backward()
    optimizer.step()
    print(f"RewardNet offline training done (MSE={loss_r.item():.4f})")


def make_env():
    base = environment.GridMazeEnv(render_mode=True, difficulty=1.0, load_maze=True, save_maze=False)
    learned = RewardEnv.LearnedRewardEnv(base, r_model, optimizer, expert_states, discriminator=discriminator,
                                         buffer_size=512, device='cpu')

    return RewardEnv.ExpertActionWrapper(learned, EXPERT_ACTIONS, n_forced=5)


vec_env = DummyVecEnv([make_env])

mem = vec_env.envs[0].env.memory
bad_cb = environment.BadActionCallback(mem, batch_size=128, lr=1e-4)

# 1. Initialize PPO agent
model = PPO("MlpPolicy", vec_env,
            n_steps=1024, batch_size=512,
            learning_rate=3e-4, gamma=0.9,
            gae_lambda=0.95, clip_range=0.1,
            ent_coef=0.8, verbose=1, device=Device)

expert_obs_t = expert_obs_tensor
expert_act_t = expert_actions_tensor

# 创建有效步掩码（假设无效步用0填充）
valid_mask = (expert_actions != 0)  # [num_traj, max_len]

# 创建DataLoader（按轨迹批量加载）
batch_size = 32  # 每次处理32条轨迹
expert_dataset = ExpertDataset(expert_obs, expert_actions, valid_mask)

expert_loader = torch.utils.data.DataLoader(
    expert_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=lambda batch: (
        torch.stack([item[0] for item in batch]),  # [B, L, 8]
        torch.stack([item[1] for item in batch]),  # [B, L]
        torch.stack([item[2] for item in batch])  # [B, L]
    ))

# 2. 修改BC训练循环（集成奖励网络）
bc_optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-4)

joint_optimizer = torch.optim.Adam(
    list(model.policy.parameters()) + list(r_model.parameters()),  # 联合参数
    lr=1e-4
)

bc_weight = 1.0  # BC损失权重
reward_weight = 0.5  # 奖励损失权重

bc_loss = []
for epoch in range(BC_EPOCHS):
    for batch_states, batch_actions, batch_mask in expert_loader:
        # 移动到设备
        batch_states = batch_states.to(Device)  # [B, L, 8]
        batch_actions = batch_actions.to(Device)  # [B, L]
        batch_mask = batch_mask.to(Device)  # [B, L]

        flat_states = batch_states[batch_mask]  # [N_valid, 8]
        flat_actions = batch_actions[batch_mask]  # [N_valid]

        noisy_states = flat_states + torch.randn_like(flat_states) * 0.05

        dist = model.policy.get_distribution(noisy_states)
        logits = dist.distribution.logits  # [N_valid, 4]

        bc_loss = F.cross_entropy(logits, flat_actions)

        actions_onehot = F.one_hot(flat_actions, num_classes=4).float()  # [N_valid, 4]

        pred_rewards = r_model(flat_states, actions_onehot)  # [N_valid, 1]

        reward_loss = pred_rewards.mean()

        total_loss = bc_weight * bc_loss + reward_weight * reward_loss

        joint_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.3)
        torch.nn.utils.clip_grad_norm_(r_model.parameters(), 0.5)  # 防止梯度爆炸
        joint_optimizer.step()

    print(f"Epoch {epoch + 1}: Total Loss={total_loss.item():.10f} "
          f"(BC={bc_loss.item():.10f}, Reward={reward_loss.item():.10f})")

# Initialize Curiosity module
curiosity = Network.CuriosityModule(state_dim=8, action_dim=4, hidden_dim=256).to(Device)
curiosity_optimizer = torch.optim.Adam(curiosity.parameters(), lr=1e-3)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


r_model.apply(weights_init)
discriminator.apply(weights_init)


class IRLCallback(BaseCallback):
    def __init__(self, r_model, discriminator, curiosity, curiosity_optimizer, expert_dataset, memory,
                 bc_weight=1.0, reward_weight=0.5, verbose=0):
        super().__init__(verbose)
        self.r_model = r_model
        self.discriminator = discriminator
        self.curiosity = curiosity
        self.expert_dataset = expert_dataset
        self.memory = memory
        self.bc_weight = bc_weight
        self.reward_weight = reward_weight
        self.iter_count = 0
        self.curiosity_optimizer = curiosity_optimizer

        self.expert_loader = torch.utils.data.DataLoader(
            expert_dataset, batch_size=32, shuffle=True,
            collate_fn=lambda batch: (
                torch.stack([item[0] for item in batch]),  # states
                torch.stack([item[1] for item in batch]),  # actions
                torch.stack([item[2] for item in batch])  # masks
            )
        )
        # 初始化绘图数据
        self.iterations = []
        self.reward_values = []
        self.loss_values = []
        self.similarity_values = []
        self.total_rewards_values = []
        self.diver_loss_values = []

        # # 初始化绘图界面
        # plt.ion()
        # self.fig, (
        #     # self.reward_ax,
        #     # self.loss_ax,
        #     # self.similarity_ax,
        #     self.total_rewards_ax,
        #     self.diver_ax
        # ) = plt.subplots(2, 1, figsize=(8, 12))
        # self.fig.subplots_adjust(hspace=0.5)
        #
        # # self.reward_line, = self.reward_ax.plot([], [], label="Curiosity Loss", color="blue")
        # # self.loss_line, = self.loss_ax.plot([], [], label="Discriminator Loss", color="red")
        # # self.similarity_line, = self.similarity_ax.plot([], [], label="Similarity Loss", color="green")
        # self.total_rewards_line, = self.total_rewards_ax.plot([], [], label="Total Rewards", color="purple")
        # self.diver_loss_line, = self.diver_ax.plot([], [], label="Diversity Loss", color="orange")

        # 设置坐标轴标签
        # self.reward_ax.set_xlabel("Iterations")
        # self.reward_ax.set_ylabel("Curiosity Loss")
        # self.loss_ax.set_xlabel("Iterations")
        # self.loss_ax.set_ylabel("Discriminator Loss")
        # self.similarity_ax.set_xlabel("Iterations")
        # self.similarity_ax.set_ylabel("Similarity Loss")
        # self.total_rewards_ax.set_xlabel("Iterations")
        # self.total_rewards_ax.set_ylabel("Total Rewards")
        # self.diver_ax.set_xlabel("Iterations")
        # self.diver_ax.set_ylabel("Diversity Loss")
        #
        # # self.reward_ax.legend()
        # # self.loss_ax.legend()
        # # self.similarity_ax.legend()
        # self.total_rewards_ax.legend()
        # self.diver_ax.legend()

        # 数据缓存
        self.curiosity_loss = 0.0
        self.discriminator_loss = 0.0
        self.similarity_reward = 0.0
        self.total_reward = 0.0
        self.diver_loss = 0.0

    def _on_step(self) -> bool:
        self.iter_count += 1

        # 每 100 步更新奖励网络和判别器
        if self.iter_count % 100 == 0:
            for expert_states, expert_actions, expert_masks in self.expert_loader:
                expert_states = expert_states.to(Device)
                expert_actions = expert_actions.to(Device)
                # 获取 agent 轨迹
                agent_states, agent_actions = self._get_agent_trajectory()
                # 更新奖励网络和判别器
                train_reward_network(self.r_model, self.discriminator,
                                     expert_states, expert_actions,
                                     agent_states, agent_actions, optimizer)
                train_discriminator(self.discriminator, expert_states, expert_actions,
                                    agent_states, agent_actions, optimizer_D)

        # 每 60 步更新记忆库中的 bad actions
        if self.iter_count % 60 == 0:
            self._update_memory()

        # 每 100 步更新好奇心模块
        if self.iter_count % 100 == 0:
            self._update_curiosity()

        # 更新指标
        self.iterations.append(self.iter_count)
        self.reward_values.append(self.curiosity_loss)  # 确保是标量
        self.loss_values.append(self.discriminator_loss)  # 确保是标量
        self.similarity_values.append(self.similarity_reward)  # 确保是标量
        self.total_rewards_values.append(self.total_reward)  # 确保是标量
        self.diver_loss_values.append(self.diver_loss)  # 确保是标量

        # if self.iter_count % 10 == 0:
        #     print(
        #         f"IRL iteration {self.iter_count}/{IRL_ITERS}",
        #         # f"Discriminator loss: {self.loss_values[-1]:.5f}",
        #         # f"Curiosity loss: {self.reward_values[-1]:.5f}",
        #         # f"Total reward: {self.total_rewards_values[-1]:.5f}",
        #         # f"Similarity reward: {self.similarity_values[-1]:.8f}",
        #         # f" Diversity loss: {self.diver_loss:.5f}",
        #         f"Reward_net loss: {loss_r:.10f}, BC loss: {bc_loss.item():.10f}",
        #     )

        # 保存数据文件
        np.savetxt('reward_values.txt', np.array(self.reward_values), fmt='%.4f')
        np.savetxt('loss_values.txt', np.array(self.loss_values), fmt='%.4f')
        np.savetxt('similarity_values.txt', np.array(self.similarity_values), fmt='%.4f')

        # 保存模型和专家轨迹
        self.model.save("ppo_grid_maze_irl_tk")

        # save_expert(self.model.get_env().envs[0], path="expert_traj.npz", expert_actions=EXPERT_ACTIONS)

        # 更新图表
        # self._update_plots()
        return True

    # def _update_plots(self):
    #     # 更新每条曲线
    #     # self.reward_line.set_data(self.iterations, self.reward_values)
    #     # self.loss_line.set_data(self.iterations, self.loss_values)
    #     # self.similarity_line.set_data(self.iterations, self.similarity_values)
    #     self.total_rewards_line.set_data(self.iterations, self.total_rewards_values)
    #     self.diver_loss_line.set_data(self.iterations, self.diver_loss_values)
    #
    #     # 自动缩放坐标轴
    #     for ax in [
    #         # self.reward_ax, self.loss_ax, self.similarity_ax,
    #                self.total_rewards_ax, self.diver_ax]:
    #         ax.relim()
    #         ax.autoscale_view()
    #
    #     # 刷新图形
    #     self.fig.canvas.draw()
    #     self.fig.canvas.flush_events()
    #     plt.pause(0.01)

    def _get_agent_trajectory(self):
        env = self.model.get_env().envs[0]  # 获取第一个环境

        states, actions = [], []
        obs, _ = env.reset()

        for _ in range(512):
            action, _ = self.model.predict(obs, deterministic=True)
            next_obs, reward, done, trunc, info = env.step(action)
            states.append(obs)
            actions.append(action)
            obs = next_obs
            if done or trunc:
                break

        # Convert states to tensor
        states_tensor = torch.tensor(np.array(states), device=Device)

        # Handle actions more carefully
        if len(actions) == 0:
            # Create empty tensor with correct dimensions if no actions
            actions_tensor = torch.tensor([], device=Device)
        else:
            # Convert actions to numpy array first to ensure consistency
            actions_np = np.array(actions)
            actions_tensor = torch.tensor(actions_np, device=Device)

        return states_tensor, actions_tensor

    def _update_memory(self):
        # Use the appropriate size attribute instead of len()
        # Common options in RL memory buffers are:

        # Option 1 - Try checking memory.size
        if hasattr(self.memory, 'size') and self.memory.size < 64:
            return

        # Option 2 - Try checking if memory buffer is empty using a safer approach
        try:
            samples, indices, weights = self.memory.sample(64)
        except (ValueError, IndexError) as e:
            # Memory doesn't have enough samples
            print(f"Skipping memory update: {e}")
            return

        states, actions, bad_flags = zip(*samples)
        states_tensor = torch.tensor(np.array(states), device=Device)
        actions_tensor = torch.tensor(actions, device=Device)
        dist = self.model.policy.get_distribution(states_tensor)
        log_probs = dist.log_prob(actions_tensor)
        bc_loss = -log_probs.mean()
        bc_loss.backward()
        self.model.policy.optimizer.step()
        self.model.policy.optimizer.zero_grad()
        with torch.no_grad():
            new_dist = self.model.policy.get_distribution(states_tensor)
            new_log_probs = new_dist.log_prob(actions_tensor)
            errors = (new_log_probs - log_probs).abs().cpu().numpy()
        self.memory.update_priorities(indices, errors)

    def _update_curiosity(self):
        env = self.model.get_env().envs[0]
        states, actions = [], []
        obs, _ = env.reset()
        for _ in range(512):
            action, _ = self.model.predict(obs, deterministic=True)
            next_obs, reward, done, trunc, info = env.step(action)
            states.append(obs)
            actions.append(action)
            obs = next_obs
            if done or trunc:
                break

        if len(states) <= 1:
            print("Not enough state transitions for curiosity update")
            return

        # Check if actions exist and convert safely to numpy first
        if not actions:
            print("No actions collected for curiosity update")
            return

        # Convert to numpy arrays for consistency
        states_np = np.array(states)
        actions_np = np.array(actions)

        # Then convert to tensors
        states_tensor = torch.tensor(states_np, device=Device)
        actions_tensor = torch.tensor(actions_np, device=Device)

        intrinsic_rewards = []
        for i in range(len(states_tensor) - 1):
            current_state = states_tensor[i].unsqueeze(0)
            next_state = states_tensor[i + 1].unsqueeze(0)
            action = actions_tensor[i].unsqueeze(0)
            intrinsic_reward, _, _ = self.curiosity(current_state, next_state, action)
            intrinsic_rewards.append(intrinsic_reward)

        if not intrinsic_rewards:
            return

        total_forward_loss = torch.mean(torch.stack(intrinsic_rewards))
        self.curiosity_optimizer.zero_grad()
        total_forward_loss.backward()
        self.curiosity_optimizer.step()


irl_callback = IRLCallback(r_model, discriminator, curiosity, curiosity_optimizer, expert_dataset,
                           vec_env.envs[0].env.memory)
bad_cb = environment.BadActionCallback(vec_env.envs[0].env.memory, batch_size=128, lr=1e-4)

model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[irl_callback, bad_cb],
    tb_log_name="IRL_PPO"
)

save_ppo_trajectories(vec_env.envs[0], best_path="ppo_best_traj.npz", all_success_path="all_successful_ppo.npz",
                      n_trials=10000, model=model)

mean, std = evaluate_policy(model, environment.GridMazeEnv(render_mode=False),
                            n_eval_episodes=50)
print(f"\nMean TRUE return over 50 eval episodes: {mean:.2f} ± {std:.2f}")
