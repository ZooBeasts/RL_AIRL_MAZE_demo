
import tkinter as tk
import json
import numpy as np
import os
from environment import GridMazeEnv

class ExpertMazeEditor:
    def __init__(self, master, maze_file, cell_size=20):
        self.cell = cell_size
        self.master = master
        self._load_maze(maze_file)
        self.all_paths = []
        self.all_actions = []
        self.path = []
        self.actions = []

        size = self.size * self.cell
        self.canvas = tk.Canvas(master, width=size, height=size, bg="white")
        self.canvas.pack()
        self._draw_grid()
        self.agent = None

        btn_frame = tk.Frame(master)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Again", command=self._on_again).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Finish", command=self._on_finish).pack(side=tk.LEFT, padx=5)
        self.canvas.bind("<Button-1>", self._on_click)

    def _load_maze(self, maze_file):
        if not os.path.isfile(maze_file):
            maze_file = GridMazeEnv.MAZE_FILE
            print(f"Warning: `{maze_file}` not found, loading default maze `{maze_file}`")
        with open(maze_file) as f:
            hell_list = json.load(f)
        self.hell = set(tuple(x) for x in hell_list)
        self.goal = GridMazeEnv.GOAL
        self.size = GridMazeEnv.SIZE

    def _draw_grid(self):
        for i in range(self.size +1):
            x = i* self.cell
            self.canvas.create_line(x, 0, x, self.size * self.cell)
            self.canvas.create_line(0, x, self.size * self.cell, x)
        for x, y in self.hell:
            self.canvas.create_rectangle(
                x * self.cell + 2, y * self.cell + 2,
                (x + 1) * self.cell - 2, (y + 1) * self.cell - 2,
                fill="black"
            )
        gx, gy = self.goal
        self.canvas.create_oval(
            gx * self.cell + 20, gy * self.cell + 20,
            (gx + 1) * self.cell - 20, (gy + 1) * self.cell - 20,
            fill="yellow"
        )

    def _check_surroundings(self, x, y):
        # Check the four cardinal directions for obstacles
        # Returns [up, down, left, right] where 1 means obstacle, 0 means clear
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        surroundings = []

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Check if out of bounds or is a hell cell
            if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size or (nx, ny) in self.hell:
                surroundings.append(1)  # Obstacle
            else:
                surroundings.append(0)  # Clear

        return surroundings

    def _on_click(self, event):
        cx, cy = event.x // self.cell, event.y // self.cell
        gx, gy = self.goal  # Get goal position

        # Get surrounding state (up, down, left, right obstacles)
        surroundings = self._check_surroundings(cx, cy)

        # Create 8-dimensional state: [x, y, gx, gy, up, down, left, right]
        current_state = [cx, cy, gx, gy] + surroundings

        if not self.path:
            self.path.append(current_state)
        else:
            px, py = self.path[-1][0], self.path[-1][1]  # Get position from previous state
            dx, dy = cx - px, cy - py
            mapping = {(0, 1): 1, (0, -1): 0, (1, 0): 3, (-1, 0): 2}
            action = mapping.get((dx, dy))
            if action is None:
                return
            self.actions.append(action)
            self.path.append(current_state)
        self._draw_agent(cx, cy)

        if (cx, cy) in self.hell:
            return  # Skip trap area

    def _draw_agent(self, x, y):
        if self.agent:
            self.canvas.delete(self.agent)
        pad = self.cell // 3
        self.agent = self.canvas.create_rectangle(
            x * self.cell + pad, y * self.cell + pad,
            (x + 1) * self.cell - pad, (y + 1) * self.cell - pad,
            fill="red"
        )

    def _on_again(self):
        if self.path and self.actions:
            self.all_paths.append(self.path.copy())
            self.all_actions.append(self.actions.copy())
        self.path.clear()
        self.actions.clear()
        self.canvas.delete("all")
        self._draw_grid()
        self.agent = None

    def _on_finish(self):
        # Include the last path if any
        if self.path and self.actions:
            self.all_paths.append(self.path)
            self.all_actions.append(self.actions)

        all_data_r = []
        for path in self.all_paths:
            rewards = []
            for state in path:
                x, y = state[0], state[1]
                gx, gy = state[2], state[3]
                # Custom reward rules (should match the environment)
                if (x, y) == (gx, gy):
                    r = 10.0  # Reached goal
                elif (x, y) in self.hell:
                    r = -1.0  # Hit trap
                else:
                    r = -0.01  # Step penalty
                rewards.append(r)

            all_data_r.append(np.array(rewards))

        # Save with rewards
        np.savez("expert_set.npz",
                 states=np.array(self.all_paths, dtype=object),
                 actions=np.array(self.all_actions, dtype=object),
                 rewards=np.array(all_data_r, dtype=object))

        # Also save maze layout
        with open("saved_maze_1.json", "w") as f:
            json.dump(list(self.hell), f)
        print("Saved expert_set.npz and saved_maze.json")
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Expert Path Editor")
    ExpertMazeEditor(root, maze_file='saved_maze_1.json')
    root.mainloop()




































# import tkinter as tk
# import json
# import numpy as np
# import os
# from environment import GridMazeEnv
#
#
#
# class ExpertMazeEditor:
#     def __init__(self, master, maze_file, cell_size=20):
#         self.cell = cell_size
#         self.master = master
#         self._load_maze(maze_file)
#         self.all_paths = []
#         self.all_actions = []
#         self.path = []
#         self.actions = []
#
#         size = self.size * self.cell
#         self.canvas = tk.Canvas(master, width=size, height=size, bg="white")
#         self.canvas.pack()
#         self._draw_grid()
#         self.agent = None
#
#         btn_frame = tk.Frame(master)
#         btn_frame.pack(pady=5)
#         tk.Button(btn_frame, text="Again", command=self._on_again).pack(side=tk.LEFT, padx=5)
#         tk.Button(btn_frame, text="Finish", command=self._on_finish).pack(side=tk.LEFT, padx=5)
#         self.canvas.bind("<Button-1>", self._on_click)
#
#     def _load_maze(self, maze_file):
#         if not os.path.isfile(maze_file):
#             maze_file = GridMazeEnv.MAZE_FILE
#             print(f"Warning: `{maze_file}` not found, loading default maze `{maze_file}`")
#         with open(maze_file) as f:
#             hell_list = json.load(f)
#         self.hell = set(tuple(x) for x in hell_list)
#         self.goal = GridMazeEnv.GOAL
#         self.size = GridMazeEnv.SIZE
#
#     def _draw_grid(self):
#         for i in range(self.size+1):
#             x = i*self.cell
#             self.canvas.create_line(x, 0, x, self.size*self.cell)
#             self.canvas.create_line(0, x, self.size*self.cell, x)
#         for x, y in self.hell:
#             self.canvas.create_rectangle(
#                 x*self.cell+2, y*self.cell+2,
#                 (x+1)*self.cell-2, (y+1)*self.cell-2,
#                 fill="black"
#             )
#         gx, gy = self.goal
#         self.canvas.create_oval(
#             gx*self.cell+20, gy*self.cell+20,
#             (gx+1)*self.cell-20, (gy+1)*self.cell-20,
#             fill="yellow"
#         )
#
#     def _on_click(self, event):
#         cx, cy = event.x // self.cell, event.y // self.cell
#         gx, gy = self.goal  # 获取目标位置
#         current_state = [cx, cy, gx, gy]  # 4维状态
#         # current_state = [cx / (self.size - 1), cy / (self.size - 1),
#         #                  gx / (self.size - 1), gy / (self.size - 1)]
#
#
#
#         if not self.path:
#             self.path.append(current_state)
#         else:
#             px, py, _, _ = self.path[-1]  # 获取前一个状态的位置
#             dx, dy = cx - px, cy - py
#             mapping = {(0, 1): 1, (0, -1): 0, (1, 0): 3, (-1, 0): 2}
#             action = mapping.get((dx, dy))
#             if action is None:
#                 return
#             self.actions.append(action)
#             self.path.append(current_state)  # 保存完整状态
#         self._draw_agent(cx, cy)
#
#         if (cx, cy) in self.hell:
#             return  # 跳过陷阱区域
#
#
#
#     def _draw_agent(self, x, y):
#         if self.agent:
#             self.canvas.delete(self.agent)
#         pad = self.cell//3
#         self.agent = self.canvas.create_rectangle(
#             x*self.cell+pad, y*self.cell+pad,
#             (x+1)*self.cell-pad, (y+1)*self.cell-pad,
#             fill="red"
#         )
#
#     def _on_again(self):
#         if self.path and self.actions:
#             self.all_paths.append(self.path.copy())
#             self.all_actions.append(self.actions.copy())
#         self.path.clear()
#         self.actions.clear()
#         self.canvas.delete("all")
#         self._draw_grid()
#         self.agent = None
#
#     def _on_finish(self):
#         # include the last if any
#         if self.path and self.actions:
#             self.all_paths.append(self.path)
#             self.all_actions.append(self.actions)
#
#         all_data_r = []
#         for path in self.all_paths:
#             rewards = []
#             for (x, y, gx, gy) in path:  # 假设状态为4维[x,y,gx,gy]
#                 # 自定义奖励规则（需与环境一致）
#                 if (x, y) == (gx, gy):
#                     r = 10.0  # 到达目标
#                 elif (x, y) in self.hell:
#                     r = -1.0  # 触碰陷阱
#                 else:
#                     r = -0.01  # 每步惩罚
#                 rewards.append(r)
#
#
#             all_data_r.append(np.array(rewards))
#
#         # 保存时包含奖励
#         np.savez("expert_set.npz",
#                  states=np.array(self.all_paths, dtype=object),
#                  actions=np.array(self.all_actions, dtype=object),
#                  rewards=np.array(all_data_r, dtype=object))
#
#         # also save maze layout
#         with open("saved_maze_1.json", "w") as f:
#             json.dump(list(self.hell), f)
#         print("Saved expert_set.npz and saved_maze.json")
#         self.master.destroy()
#
# if __name__ == "__main__":
#     root = tk.Tk()
#     root.title("Expert Path Editor")
#     ExpertMazeEditor(root, maze_file='saved_maze_1.json')
#     root.mainloop()