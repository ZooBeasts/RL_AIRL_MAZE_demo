
import numpy as np
import matplotlib.pyplot as plt
import torch
import environment


def plot_trajectories(expert_states, agent_states):
    plt.figure(figsize=(10, 5))
    plt.plot(expert_states[:, 0], expert_states[:, 1], label="Expert", color="blue", marker="o")
    plt.plot(agent_states[:, 0], agent_states[:, 1], label="Agent", color="red", marker="x")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory Comparison")
    plt.show()


# def save_expert(env, path=None, expert_actions=None):
#     all_states, all_actions = [], []
#     max_len = max(len(states) for states in all_states)
#
#     # Pad states (assuming they are 2D arrays)
#     padded_states = [np.pad(states, ((0, max_len - len(states)), (0, 0)), mode='constant')
#                      for states in all_states
#                      ]
#
#     # Pad actions (assuming they are 1D arrays)
#     padded_actions = [np.pad(actions, (0, max_len - len(actions)), mode='constant')
#                       for actions in all_actions]
#     for expert_actions in expert_actions:
#         obs, _ = env.reset()
#         states, actions = [], []
#         for a in expert_actions:
#             states.append(obs)
#             actions.append(a)
#             obs, r, term, trunc, _ = env.step(a)
#             if term or trunc:
#                 break
#
#         all_states.append(np.array(states, dtype=np.float32))
#         all_actions.append(np.array(actions, dtype=np.int32))
#
#     np.savez(path, states=np.array(padded_states), actions=np.array(padded_actions))
def save_ppo_trajectories(env, best_path="ppo_best_traj.npz", all_success_path="all_successful_ppo.npz",
                          n_trials=10000, model=None):
    if model is None:
        from main import model

    # Use provided environment if possible
    use_env = env
    needs_cleanup = False

    # Create non-rendering env only if needed
    if getattr(env, "render_mode", True):
        use_env = environment.GridMazeEnv(render_mode=False)
        needs_cleanup = True

    best_return = -1e9
    best_pair = None
    successful_paths = []

    # Run all trials
    for i in range(n_trials):
        obs, _ = use_env.reset()
        states, actions = [], []
        total_return = 0.0
        done = False
        reached_exit = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            states.append(obs)
            actions.append(int(action))
            obs, r, term, trunc, _ = use_env.step(action)
            total_return += r
            done = term or trunc
            if r == 1.0:
                reached_exit = True

        # Check if this is the best trajectory (only consider first 10 trials)
        if i < 10 and total_return > best_return:
            best_return = total_return
            best_pair = (states, actions)

        # Save successful trajectories
        if reached_exit:
            successful_paths.append({
                "states": np.array(states, dtype=np.float32),
                "actions": np.array(actions, dtype=np.int32)
            })

    if needs_cleanup:
        use_env.close()

    # Save best trajectory
    if best_pair:
        states, actions = best_pair
        np.savez(best_path,
                 states=np.array(states, dtype=np.float32),
                 actions=np.array(actions, dtype=np.int32),
                 total_return=best_return)
        print(f"Best PPO trajectory (return={best_return:.2f}) saved to {best_path}")

    # Path deduplication and save all successful paths
    unique_paths = []
    seen_hashes = set()
    for path_item in successful_paths:
        hash_key = tuple(map(lambda x: tuple(x.round(5)), path_item["states"]))
        if hash_key not in seen_hashes:
            seen_hashes.add(hash_key)
            unique_paths.append(path_item)

    np.savez(all_success_path, paths=unique_paths)
    print(f"Saved {len(unique_paths)} unique successful paths to {all_success_path}")

    return best_return, len(unique_paths)


def simulate_expert_trajectories(expert_actions, env, max_episodes=10):
    for episode in range(max_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # 智能体动作（可忽略）
            next_obs, reward, done, trunc, info = env.step(action)



class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, states, actions, valid_mask):
        """
        states: [num_traj, max_len, 8]
        actions: [num_traj, max_len]
        valid_mask: [num_traj, max_len]
        """
        # Handle states
        if isinstance(states, torch.Tensor):
            self.states = states
        elif isinstance(states, np.ndarray):
            self.states = torch.tensor(states, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported type for states: {type(states)}")

        # Handle actions
        if isinstance(actions, torch.Tensor):
            self.actions = actions
        elif isinstance(actions, np.ndarray):
            self.actions = torch.tensor(actions, dtype=torch.long)
        else:
            raise TypeError(f"Unsupported type for actions: {type(actions)}")

        # Handle valid_mask
        if isinstance(valid_mask, torch.Tensor):
            self.valid_mask = valid_mask
        elif isinstance(valid_mask, np.ndarray):
            self.valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
        else:
            raise TypeError(f"Unsupported type for valid_mask: {type(valid_mask)}")

    def __len__(self):
        return self.states.size(0)  # 轨迹数量

    def __getitem__(self, idx):
        return (
            self.states[idx],  # [max_len, 8]
            self.actions[idx],  # [max_len]
            self.valid_mask[idx]  # [max_len]
        )




def calculate_path_diversity(paths):
    unique_states = set(tuple(s.tolist()) for s in paths)
    return len(unique_states) / (environment.GridMazeEnv.SIZE ** 2)
