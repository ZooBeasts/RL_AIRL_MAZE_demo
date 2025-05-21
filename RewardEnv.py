import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn
# import Network
import torch.nn.functional as F
import environment




ONLINE_BATCH = 64  # minibatch for on‑line updates # 0‑U 1‑D 2‑L 3‑R


class LearnedRewardEnv(gym.Wrapper):
    def __init__(self, env, reward_net, opt,expert_states, discriminator, # 预训练的奖励网络
                 buffer_size=ONLINE_BATCH, device="cpu"):
        super().__init__(env)
        self.r_model, self.optimizer, self.buf, self.device = reward_net, opt, [], device
        self.B = buffer_size
        self.alpha = 1.0
        self.step_count = 0
        self.alpha_decay_steps = 50000
        self.last_actions = []
        self.bad_states = set()
        self.sa_counts = {}
        self.beta_novelty = 10.0
        self.action_penalty = 0.1
        self.epsilon = 0.6  # 初始探索概率
        self.epsilon_decay = 0.9999  # 衰减系数
        self.min_epsilon = 0.05
        self.memory = MemoryBuffer(1000)
        self.expert_states = expert_states
        self.discriminator = discriminator

    def reset(self, **kw):
        self.episode_states = []
        self.last_actions = []  # 如果需要重置，确保逻辑正确
        return super().reset(**kw)


    def step(self, action):
        obs, r_true, term, trunc, info = self.env.step(action)

        # 归一化状态
        state_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 格式化动作为 one-hot
        if isinstance(action, (int, np.int32, np.int64)):
            action_int = int(action)
            action_tensor = torch.zeros(1, 4, device=self.device)
            action_tensor[0, action_int] = 1.0
        else:
            action_tensor = F.one_hot(torch.tensor(action, device=self.device), num_classes=4).float()
            if action_tensor.dim() == 1:
                action_tensor = action_tensor.unsqueeze(0)

        # 存储经验
        self.buf.append((obs, action, r_true))
        s = tuple(obs.tolist())
        key = (s, action)

        # 更新 reward network
        if len(self.buf) >= self.B:
            o, a, r = zip(*self.buf)
            o_tensor = torch.tensor(np.array(o), dtype=torch.float32, device=self.device)
            r_tensor = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
            a_int = torch.tensor(np.array(a), dtype=torch.long, device=self.device)
            a_tensor = F.one_hot(a_int, num_classes=4).float()
            loss = nn.MSELoss()(self.r_model(o_tensor, a_tensor), r_tensor)
            self.optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.r_model.parameters(), 0.5)
            loss.backward()
            self.optimizer.step()
            self.buf.clear()

        # 奖励混合
        with torch.no_grad():
            r_pred = self.r_model(state_tensor, action_tensor).item()

        # 内在奖励
        self.sa_counts = getattr(self, 'sa_counts', {})
        self.sa_counts[key] = self.sa_counts.get(key, 0) + 1
        intrinsic = self.beta_novelty / np.sqrt(self.sa_counts[key])

        if self.last_actions:
            action_penalty = 0.2 if action == self.last_actions[-1] else 0.0
        else:
            action_penalty = 0.0

        # 动作惩罚
        # action_penalty = 0.2 if action == self.last_actions[-1] else 0.0
        self.last_actions.append(action)
        if len(self.last_actions) > 10:
            self.last_actions.pop(0)

        # 判别器奖励
        d_logits = self.discriminator(state_tensor, action_tensor)
        r_disc = torch.sigmoid(d_logits).log().item()

        # 相似性奖励
        similarity_rewards = []
        for expert_traj in self.expert_states:  # ✅ 使用传入的 expert_states
            s_array = np.array(s)
            min_dist = min(np.linalg.norm(s_array - expert_state) for expert_state in expert_traj)
            similarity_rewards.append(np.exp(-min_dist * 0.5))
        similarity_reward = np.mean(similarity_rewards)

        # 奖励混合公式（保持原有逻辑）
        total_reward = (
                0.3 * r_true +
                0.5 * r_pred +
                0.2 * r_disc +
                0.1 * intrinsic +
                0.1 * similarity_reward -
                0.1 * action_penalty
        )

        return obs, total_reward, term, trunc, info  # ✅ 确保始终返回五元组

class ExpertActionWrapper(gym.Wrapper):
    def __init__(self, env, expert_trajs, n_forced=300):
        super().__init__(env)
        self.expert_trajs = expert_trajs
        self.n_forced = n_forced  # 强制使用专家动作的步数
        self.current_traj = None
        self.step_idx = 0
        self.episode_count = 0

    def reset(self, **kwargs):
        # 每个episode使用新专家轨迹
        # 在 ExpertActionWrapper.reset() 中添加验证
        self.current_traj = [int(a) for a in self.expert_trajs[self.episode_count % len(self.expert_trajs)]]
        # self.current_traj = self.expert_trajs[self.episode_count % len(self.expert_trajs)]
        self.episode_count += 1
        self.step_idx = 0
        return super().reset(**kwargs)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.item())

        # 动态调整 n_forced（随训练进程减少强制步数）
        decay_rate = 0.99
        self.n_forced = max(50, int(self.n_forced * decay_rate))

        if self.step_idx < self.n_forced and self.step_idx < len(self.current_traj):
            expert_action = self.current_traj[self.step_idx]
            action = expert_action
            self.step_idx += 1
        else:
            self.step_idx += 1
        return super().step(action)



def generate_expert_rollout(expert_actions):
    env = environment.GridMazeEnv(render_mode=False)
    all_data_obs, all_data_r, all_data_actions = [], [], []  # 添加 all_data_actions

    for action_seq in expert_actions:
        obs, _ = env.reset()
        data_obs, data_r, data_actions = [], [], []

        for a in action_seq:
            obs2, r, term, trunc, _ = env.step(a)
            data_obs.append(obs)
            data_r.append(r)
            data_actions.append(a)  # 记录动作
            obs = obs2
            if env.pos in env.HELL:  # 检查是否进入障碍物
                print(f"Expert action {a} led to HELL at {env.pos}")

            if term or trunc:
                break

        all_data_obs.append(np.array(data_obs))
        all_data_r.append(np.array(data_r))
        all_data_actions.append(np.array(data_actions))  # 添加动作序列

    env.close()
    return all_data_obs, all_data_actions, all_data_r  # 返回三个值


def collect_agent_trajectories(model, env, max_steps=512):
    states, actions, rewards = [], [], []
    obs, _ = env.reset()
    if hasattr(env, "HELL"):
        hell = env.HELL
    elif hasattr(env, "env") and hasattr(env.env, "HELL"):
        hell = env.env.HELL
    else:
        hell = set()

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)  # Convert to int if necessary
        next_obs, reward, done, trunc, info = env.step(action)
        if tuple(obs.tolist()) in hell:
            obs = next_obs if not (done or trunc) else env.reset()[0]
            continue  # Skip HELL state
        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        obs = next_obs if not (done or trunc) else env.reset()[0]
    return np.array(states), np.array(actions), np.array(rewards)


def train_ppo_with_rewards(model, states, actions, bonus_rewards):
    model.policy.train()
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    states = states.detach()
    actions = actions.detach()
    bonus_rewards = bonus_rewards.detach()

    if not isinstance(states, torch.Tensor):
        states_tensor = torch.tensor(states, dtype=torch.float32, device=bonus_rewards.device)
    else:
        states_tensor = states.detach().clone().to(bonus_rewards.device)

    if not isinstance(actions, torch.Tensor):
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=bonus_rewards.device)
    else:
        actions_tensor = actions.detach().clone().to(bonus_rewards.device)

    # Get distribution and action log probabilities
    dist = model.policy.get_distribution(states_tensor)
    log_probs = dist.log_prob(actions_tensor)
    with torch.no_grad():
        is_bad_action = torch.zeros_like(log_probs)
        for i, s in enumerate(states):
            if tuple(s.tolist()) in model.env.envs[0].env.bad_states:
                is_bad_action[i] = 1.0

    # Use bonus rewards as advantage surrogate (detached so gradients are not backpropagated)
    advantage = bonus_rewards.detach().view(-1)

    loss = -(log_probs * advantage).mean() + 0.5 * (log_probs * is_bad_action).mean()

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()



class MemoryBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []  # 存储经验项 (state, action, bad)
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # 优先级数组
        self.timestamps = np.zeros((capacity,), dtype=np.int64)  # 时间戳数组
        self.pos = 0  # 当前插入位置
        self.full = False
        self.time_step = 0  # 全局时间步

    def sample_bad(self, batch_size):
        bad_indices = [i for i, (s, a, bad) in enumerate(self.buffer) if bad]
        indices = np.random.choice(bad_indices, batch_size)
        return [self.buffer[i] for i in indices], indices, np.ones(batch_size)

    def add(self, state, action, bad=False, error=1e-5, threshold=0.1):
        max_priority = self.priorities.max() if self.full else 1.0
        self.time_step += 1  # 更新全局时间步

        # 检查是否重复样本
        for existing_state, _, _ in self.buffer:
            if np.linalg.norm(state - existing_state) < threshold:
                return  # 相似状态已存在，跳过存储

        # 添加新样本
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, bad))
        else:
            self.buffer[self.pos] = (state, action, bad)

        self.priorities[self.pos] = (np.abs(error) + 1e-5) ** self.alpha
        self.timestamps[self.pos] = self.time_step  # 记录添加时间

        self.pos = (self.pos + 1) % self.capacity
        if self.pos >= self.capacity:
            self.full = True

    def sample(self, batch_size):
        if not self.full:
            priorities = self.priorities[:self.pos]
        else:
            priorities = self.priorities

        probs = priorities / priorities.sum()
        indices = np.random.choice(len(priorities), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # 计算重要性采样权重
        total = len(priorities)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_errors):
        for idx, error in zip(batch_indices, batch_errors):
            self.priorities[idx] = (np.abs(error) + 1e-5) ** self.alpha

    def clean(self, max_age=1000, min_priority=0.1):
        current_time = self.time_step
        indices_to_remove = []

        for i in range(len(self.buffer)):
            if self.full or i < self.pos:
                age = current_time - self.timestamps[i]
                priority = self.priorities[i]

                # 优先级低且时间久远的样本视为过时
                if priority < min_priority and age > max_age:
                    indices_to_remove.append(i)

        # 逆序删除，防止索引错位
        for i in sorted(indices_to_remove, reverse=True):
            del self.buffer[i]
            self.priorities[i] = 0
            self.timestamps[i] = 0

        # 更新 pos 和 full 标志
        if len(self.buffer) < self.capacity:
            self.full = False
        else:
            self.full = True