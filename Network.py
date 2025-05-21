import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(RewardNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Input processing layers
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)

        self.drop = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)

        # Reward head components
        self.base_reward_head = nn.Linear(hidden_dim, 1)
        self.path_reward_head = nn.Linear(hidden_dim, 1)
        self.path_length_head = nn.Linear(hidden_dim, 1)
        self.hell_penalty_head = nn.Linear(hidden_dim, 1)

    def forward(self, states, actions):
        if states.dim() == 3:
            batch_size, seq_len, state_dim = states.shape
            states_flat = states.view(-1, state_dim)  # [batch*seq, state_dim]
            if actions.dim() == 2:  # [batch, seq]
                actions_flat = actions.view(-1)  # [batch*seq]
                actions_onehot = F.one_hot(actions_flat.long(), num_classes=self.action_dim).float()  # [batch*seq, action_dim]
            elif actions.dim() == 3:  # [batch, seq, action_dim] 的 one-hot
                actions_onehot = actions.view(-1, self.action_dim)  # [batch*seq, action_dim]
            x = torch.cat([states_flat, actions_onehot.detach()], dim=1)
        else:
            if actions.dim() == 1:  # [batch]
                actions_onehot = F.one_hot(actions.long(), num_classes=self.action_dim).float()  # [batch, action_dim]
            elif actions.dim() == 2:  # [batch, action_dim] 的 one-hot
                actions_onehot = actions
            x = torch.cat([states, actions_onehot], dim=1)

        # Process through network
        x = F.relu(self.fc1(x))
        # x = self.drop(x)
        x = F.relu(self.fc2(x))
        # x = self.drop(x)
        x = self.fc3(x)

        # Generate different reward components
        base_reward = self.base_reward_head(x)
        path_reward = self.path_reward_head(x)
        path_length_reward = self.path_length_head(x)
        hell_penalty = self.hell_penalty_head(x)

        # Combine all reward components
        total_reward = base_reward + path_reward + path_length_reward + hell_penalty

        # Restore original batch shape if needed
        if states.dim() == 3:
            total_reward = total_reward.reshape(batch_size, seq_len, 1)
        return torch.sigmoid(total_reward)


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)

        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, states, actions):
        if states.dim() == 3:
            batch_size, seq_len, state_dim = states.shape
            states = states.view(-1, state_dim)  # [batch*seq, state_dim]
            if actions.dim() == 3:  # [batch, seq, action_dim] 的 one-hot
                actions = actions.view(-1, self.action_dim)  # [batch*seq, action_dim]
            elif actions.dim() == 2:  # [batch, seq]
                actions = F.one_hot(actions.view(-1).long(),
                                    num_classes=self.action_dim).float()  # [batch*seq, action_dim]
        else:
            if actions.dim() == 1:  # [batch]
                actions = F.one_hot(actions.long(), num_classes=self.action_dim).float()  # [batch, action_dim]
            elif actions.dim() == 2:  # [batch, action_dim] 的 one-hot
                actions = actions

        # 编码状态和动作
        state_features = self.state_encoder(states)
        action_features = self.action_encoder(actions)

        # 拼接并输出
        combined = torch.cat([state_features, action_features], dim=1)
        return self.discriminator(combined)


class CuriosityModule(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        # Inverse model: predicts action given current and next state features
        self.inverse_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Forward model: predicts next state feature given current state and action
        self.forward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        self.register_buffer('moving_avg', torch.tensor(1.0))

    def forward(self, state, next_state, action):
        state = state.detach()
        next_state = next_state.detach()
        action = action.detach()
        # Inverse loss
        cat = torch.cat([state, next_state], dim=-1)
        pred_action_logits = self.inverse_net(cat)
        inverse_loss = F.cross_entropy(pred_action_logits, action)

        # One-hot encode action for forward model
        action_onehot = F.one_hot(action, num_classes=pred_action_logits.shape[-1]).float()
        forward_input = torch.cat([state, action_onehot], dim=-1)
        pred_next_state = self.forward_net(forward_input)
        forward_loss = F.mse_loss(pred_next_state, next_state)

        # Intrinsic reward based on forward error
        forward_error = F.mse_loss(pred_next_state, next_state, reduction='none').mean(dim=-1)
        # intrinsic_reward = 0.1 * (forward_error / (forward_error.detach().mean() + 1e-8))  # 动态缩放

        self.moving_avg = 0.9 * self.moving_avg + 0.1 * forward_error.detach().mean()
        intrinsic_reward = 5.0 * (forward_error / (self.moving_avg + 1e-8))  # 提高beta_novelty到5.0

        return intrinsic_reward, forward_loss, inverse_loss


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
