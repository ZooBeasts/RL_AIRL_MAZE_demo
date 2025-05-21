import torch
import torch.nn.functional as F




def train_reward_network(r_model, discriminator, expert_states, expert_actions, agent_states, agent_actions, optimizer ):
    # 展平数据
    expert_states_flat = expert_states.view(-1, expert_states.size(-1))  # [B*seq, state_dim]
    expert_actions_flat = expert_actions.view(-1)  # [B*seq]
    agent_states_flat = agent_states.view(-1, agent_states.size(-1))
    agent_actions_flat = agent_actions.view(-1)

    # 专家动作one-hot编码
    expert_actions_onehot = F.one_hot(expert_actions_flat.long(), num_classes=4).float()
    agent_actions_onehot = F.one_hot(agent_actions_flat.long(), num_classes=4).float()

    # 判别器输出（专家和智能体）
    d_expert = discriminator(expert_states_flat, expert_actions_onehot)
    d_agent = discriminator(agent_states_flat, agent_actions_onehot)

    # 奖励网络目标：最大化专家轨迹的判别器输出，最小化智能体轨迹
    reward_expert = d_expert.sigmoid().log()  # 转换为奖励
    reward_agent = d_agent.sigmoid().log()

    # 使用奖励网络（r_model）预测奖励
    pred_expert = r_model(expert_states_flat, expert_actions_onehot)
    pred_agent = r_model(agent_states_flat, agent_actions_onehot)

    # 训练奖励网络：最小化预测奖励与判别器输出的差异
    loss_r = F.mse_loss(pred_expert, reward_expert) + F.mse_loss(pred_agent, reward_agent)

    optimizer.zero_grad()
    loss_r.backward()
    optimizer.step()
    return loss_r.item()

def train_discriminator(discriminator, expert_states, expert_actions, agent_states, agent_actions, optimizer_D):
    # 展平数据
    expert_states_flat = expert_states.view(-1, expert_states.size(-1))
    expert_actions_flat = expert_actions.view(-1)
    agent_states_flat = agent_states.view(-1, agent_states.size(-1))
    agent_actions_flat = agent_actions.view(-1)

    # 专家动作one-hot编码
    expert_actions_onehot = F.one_hot(expert_actions_flat.long(), num_classes=4).float()
    agent_actions_onehot = F.one_hot(agent_actions_flat.long(), num_classes=4).float()

    # 计算判别器损失
    d_expert = discriminator(expert_states_flat, expert_actions_onehot)
    d_agent = discriminator(agent_states_flat, agent_actions_onehot)

    d_expert = (d_expert - d_expert.min()) / (d_expert.max() - d_expert.min())
    d_agent = (d_agent - d_agent.min()) / (d_agent.max() - d_agent.min())


    loss_D = (d_expert.mean() - d_agent.mean())  # WGAN-GP损失
    print("d_expert.mean():", d_expert.mean().item())
    print("d_agent.mean():", d_agent.mean().item())

    # 梯度惩罚
    lambda_gp = 10
    # lambda_gp = 10 * ( 1 - ite / IRL_ITERS)
    loss_D += lambda_gp * gradient_penalty(
        expert_states_flat, expert_actions_onehot,
        agent_states_flat, agent_actions_onehot,
        discriminator
    )

    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()
    return loss_D.item()




def gradient_penalty(real_states, real_actions_onehot, fake_states, fake_actions_onehot, discriminator):
    batch_size = min(real_states.shape[0], fake_states.shape[0])

    # Ensure consistent shapes for both tensors
    real_states = real_states[:batch_size]
    fake_states = fake_states[:batch_size]

    # Make sure action one-hot tensors have the same shape
    if real_actions_onehot.shape != fake_actions_onehot.shape:
        # Ensure they have the same number of classes (4)
        if real_actions_onehot.dim() != fake_actions_onehot.dim() or real_actions_onehot.shape[-1] != \
                fake_actions_onehot.shape[-1]:
            # Reshape if needed
            num_classes = 4  # Your environment has 4 actions
            real_actions_flat = real_actions_onehot.view(-1, num_classes)[:batch_size]
            fake_actions_flat = fake_actions_onehot.view(-1, num_classes)[:batch_size]
        else:
            real_actions_flat = real_actions_onehot[:batch_size]
            fake_actions_flat = fake_actions_onehot[:batch_size]
    else:
        real_actions_flat = real_actions_onehot
        fake_actions_flat = fake_actions_onehot

    # Rest of the gradient penalty calculation
    alpha = torch.rand(batch_size, 1, device=real_states.device)

    interpolated_states = alpha * real_states + (1 - alpha) * fake_states
    interpolated_states.requires_grad_(True)

    # Make sure alpha is properly broadcast for action dimensions
    if real_actions_flat.dim() > 2:
        alpha = alpha.unsqueeze(-1)

    interpolated_actions = alpha * real_actions_flat + (1 - alpha) * fake_actions_flat

    disc_interpolates = discriminator(interpolated_states, interpolated_actions)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolated_states,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty