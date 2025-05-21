
import os
import json
import random
import torch
import numpy as np
import tkinter as tk
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn.functional as F

BG, GRID, RED, YELL, BLACK = "#FFFFFF", "#C8C8C8", "#FF0000", "#FFFF00", "#000000"
cell = 50


class TKGrid:
    def __init__(self, size, hell, goal):
        self.size, self.hell, self.goal = size, hell, goal
        self.root = tk.Tk()
        self.root.title("Grid‑maze live view")
        w = h = size * cell
        self.cv = tk.Canvas(self.root, width=w, height=h, bg=BG)
        self.cv.pack()

        self._draw_static()
        self.agent = None

    def _draw_static(self):
        for i in range(self.size + 1):
            x = i * cell
            self.cv.create_line(x, 0, x, self.size * cell, fill=GRID)
            self.cv.create_line(0, x, self.size * cell, x, fill=GRID)
        for (x, y) in self.hell:  self._square(x, y, BLACK)
        gx, gy = self.goal
        cx, cy = gx * cell + cell // 2, gy * cell + cell // 2
        self.cv.create_oval(cx - 20, cy - 20, cx + 20, cy + 20, fill=YELL, width=0)

    def _square(self, x, y, color):
        pad = 10
        return self.cv.create_rectangle(x * cell + pad, y * cell + pad,
                                        x * cell + cell - pad, y * cell + cell - pad,
                                        fill=color, width=0)

    def draw_agent(self, pos):
        x, y = pos
        if self.agent is None:
            self.agent = self._square(x, y, RED)
        else:
            self.cv.coords(self.agent,
                           x * cell + 10, y * cell + 10,
                           x * cell + cell - 10, y * cell + cell - 10)
        self.root.update_idletasks()
        self.root.update()
        # self.root.after(1)  # ✅ 添加 50ms 延迟
        # self.root.after(delay, self.root.update_idletasks)
        # self.root.after(delay, self.root.update)



class GridMazeEnv(gym.Env):
    SIZE = 20

    HELL = {(2, 2), (3, 3), (2, 3), (3, 5), (1, 7), (2, 8), (1, 9), (0, 11), (1, 12), (3, 14), (4, 15),
                     (3, 15), (1, 18), (1, 17), (0, 18), (0, 17), (0, 19), (1, 19), (18, 2), (17, 3), (17, 4), (18, 5),
                     (13, 2), (12, 4), (14, 5), (9, 6), (13, 5), (9, 2), (9, 4), (7, 3), (6, 1), (5, 1), (7, 0), (8, 4),
                     (10, 5), (8, 6), (9, 5), (12, 7), (10, 10), (12, 10), (5, 11), (8, 9), (8, 10), (7, 10), (8, 13),
                     (6, 16), (11, 16), (9, 15), (5, 4), (5, 7), (5, 14), (8, 16), (10, 17), (8, 17), (15, 18),
                     (19, 19), (18, 17), (17, 15), (14, 16), (17, 17), (16, 17), (18, 14), (19, 12), (18, 9), (17, 9),
                     (16, 10), (16, 7), (16, 1), (14, 0), (5, 18), (11, 19), (4, 19), (3, 18), (12, 12), (12, 14),
                     (14, 11)
            }

    GOAL = (12, 16)
    STEP_PENALTY, MAX_EPISODE_STEPS = -0.01, 1000
    metadata = {"render_modes": ["human"], "render_fps": 1000}
    MAZE_FILE = "saved_maze.json"


    def __init__(self, *, render_mode=False, difficulty=1.0, load_maze=False, save_maze=False):
        super().__init__()
        self._save_maze()
        self.goal = self.GOAL
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0.0, 1.0, (8,), np.float32)
        self.viewer = TKGrid(self.SIZE, self.HELL, self.GOAL) if render_mode else None
        self.max_distance = np.linalg.norm(np.array([self.SIZE - 1, self.SIZE - 1]) - np.array(self.goal))

        if load_maze and os.path.exists(self.MAZE_FILE):
            self.HELL = self._load_maze()
        else:
            self.HELL = self._generate_default_hell(difficulty)
            if save_maze:
                self._save_maze()

        self.reset()

        self.Hell = self._adjust_hell(difficulty)
        global cell

    def _generate_default_hell(self, difficulty):
        full_hell = {(2, 2), (3, 3), (2, 3), (3, 5), (1, 7), (2, 8), (1, 9), (0, 11), (1, 12), (3, 14), (4, 15),
                     (3, 15), (1, 18), (1, 17), (0, 18), (0, 17), (0, 19), (1, 19), (18, 2), (17, 3), (17, 4), (18, 5),
                     (13, 2), (12, 4), (14, 5), (9, 6), (13, 5), (9, 2), (9, 4), (7, 3), (6, 1), (5, 1), (7, 0), (8, 4),
                     (10, 5), (8, 6), (9, 5), (12, 7), (10, 10), (12, 10), (5, 11), (8, 9), (8, 10), (7, 10), (8, 13),
                     (6, 16), (11, 16), (9, 15), (5, 4), (5, 7), (5, 14), (8, 16), (10, 17), (8, 17), (15, 18),
                     (19, 19), (18, 17), (17, 15), (14, 16), (17, 17), (16, 17), (18, 14), (19, 12), (18, 9), (17, 9),
                     (16, 10), (16, 7), (16, 1), (14, 0), (5, 18), (11, 19), (4, 19), (3, 18), (12, 12), (12, 14),
                     (14, 11)
        }
        hell_count = int(len(full_hell) * difficulty)
        full_hell_list = sorted(list(full_hell))
        return set(full_hell_list[:hell_count])

    def _obs(self):
        x, y = self.pos
        gx, gy = self.GOAL
        # 检查上下左右是否有障碍
        obstacles = [
            (x, y - 1) in self.HELL,  # 上
            (x, y + 1) in self.HELL,  # 下
            (x - 1, y) in self.HELL,  # 左
            (x + 1, y) in self.HELL  # 右
        ]
        return np.array([x, y, gx, gy] + obstacles, np.float32) / (self.SIZE - 1)


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = (0, 0)
        self.steps = 0
        if self.viewer: self.viewer.draw_agent(self.pos)
        return self._obs(), {}

    def step(self, action: int):
        x, y = self.pos
        old_x, old_y = x, y
        # 执行动作
        if action == 0 and y > 0:
            y -= 1
        elif action == 1 and y < self.SIZE - 1:
            y += 1
        elif action == 2 and x > 0:
            x -= 1
        elif action == 3 and x < self.SIZE - 1:
            x += 1
        else:
            # 无效动作
            reward = -1.0
            return self._obs(), reward, True, False, {}

        self.pos = (x, y)

        # 初始化 reward 为步长惩罚
        reward = self.STEP_PENALTY
        done = False
        truncated = False

        # 检查是否到达目标或障碍物
        if self.pos == self.GOAL:
            reward = 10.0
            done = True
        elif self.pos in self.HELL:
            reward = -1.0
            done = True

        # 检查是否超出最大步数
        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True

        # 更新步数
        self.steps += 1

        # 渲染
        if self.viewer:
            self.viewer.draw_agent(self.pos)

        return self._obs(), reward, done, truncated, {}


    def render(self):
        if self.viewer: self.viewer.draw_agent(self.pos)

    def close(self):
        if self.viewer: self.viewer.root.destroy()

    def _save_maze(self):
        # Save the hell set as a list of coordinates to a json file
        with open(self.MAZE_FILE, "w") as f:
            json.dump(list(self.HELL), f)

    def _load_maze(self):
        with open(self.MAZE_FILE, "r") as f:
            data = json.load(f)
        return {tuple(coord) for coord in data}

    def _generate_hell_locations(self, count):
        hell_locations = set()
        while len(hell_locations) < count:
            x, y = np.random.randint(0, self.SIZE), np.random.randint(0, self.SIZE)
            # Don't place obstacles at start or goal
            if (x, y) != (0, 0) and (x, y) != self.GOAL:
                hell_locations.add((x, y))
        return hell_locations

    @staticmethod
    def _adjust_hell(difficulty):
        full_hell = {(2, 2), (3, 3), (2, 3), (3, 5), (1, 7), (2, 8), (1, 9), (0, 11), (1, 12), (3, 14), (4, 15),
                     (3, 15), (1, 18), (1, 17), (0, 18), (0, 17), (0, 19), (1, 19), (18, 2), (17, 3), (17, 4), (18, 5),
                     (13, 2), (12, 4), (14, 5), (9, 6), (13, 5), (9, 2), (9, 4), (7, 3), (6, 1), (5, 1), (7, 0), (8, 4),
                     (10, 5), (8, 6), (9, 5), (12, 7), (10, 10), (12, 10), (5, 11), (8, 9), (8, 10), (7, 10), (8, 13),
                     (6, 16), (11, 16), (9, 15), (5, 4), (5, 7), (5, 14), (8, 16), (10, 17), (8, 17), (15, 18),
                     (19, 19), (18, 17), (17, 15), (14, 16), (17, 17), (16, 17), (18, 14), (19, 12), (18, 9), (17, 9),
                     (16, 10), (16, 7), (16, 1), (14, 0), (5, 18), (11, 19), (4, 19), (3, 18), (12, 12), (12, 14),
                     (14, 11)
                     }
        hell_count = int(len(full_hell) * difficulty)
        return set(random.sample(list(full_hell), hell_count))


class RenderCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.frames = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        env = self.training_env.envs[0]
        while hasattr(env, "env"):
            env = env.env
        if hasattr(env, "render"):
            env.render()
        return True


class BadActionCallback(BaseCallback):
    def __init__(self, memory, batch_size=32, lr=1e-5, explore_weight=0.5):
        super().__init__()
        self.memory = memory
        self.batch_size = batch_size
        self.lr = lr
        self.explore_weight = explore_weight
        self.optimizer = None
        self.penalty_weight = 2.0

    def _on_training_start(self):
        self.optimizer = torch.optim.Adam(self.model.policy.parameters(), lr=self.lr)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        if len(self.memory.buffer) < self.batch_size:
            return True

        # 采样 bad states
            # 采样 bad states
            batch, indices, weights = self.memory.sample_bad(self.batch_size)
            # 计算策略分布
            with torch.no_grad():
                dist = self.model.policy.get_distribution(states)
                log_probs = dist.log_prob(actions)
            # 计算惩罚损失
            penalty_loss = -log_probs.mean() * self.penalty_weight
            # 计算探索损失
            with torch.no_grad():
                current_actions = self.model.predict(states, deterministic=False)[0]
            explore_loss = -F.cross_entropy(dist.distribution.logits, current_actions)
            # 更新策略
            loss = penalty_loss + self.explore_weight * explore_loss
            self.optimizer.step(loss)

        # 更新策略
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 0.5)
        self.optimizer.step()

        # 更新优先级
        with torch.no_grad():
            new_dist = self.model.policy.get_distribution(states)
            new_log_probs = new_dist.log_prob(actions)
            errors = (new_log_probs - log_probs).abs().cpu().numpy()
        self.memory.update_priorities(indices, errors)

        return True

if __name__ == "__main__":
    env = GridMazeEnv(render_mode=True, difficulty=1, load_maze=False, save_maze=True)
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        env.render()
    # Keep the Tkinter window open after the simulation ends.
    if env.viewer:
        env.viewer.root.mainloop()
