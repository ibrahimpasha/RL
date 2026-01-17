"""
Script to insert Double DQN content into the notebook.
"""

from notebook_updater import (
    load_notebook, save_notebook, create_markdown_cell, 
    create_code_cell, find_cell_by_content
)


def main():
    # Load the notebook
    nb = load_notebook('reinforcement_learning_zero_to_hero.ipynb')
    
    # Find the insertion point (after the DQN section)
    search_text = "In the next section, we'll explore **Double DQN**"
    insert_index = find_cell_by_content(nb, search_text)
    
    if insert_index == -1:
        print("Error: Could not find insertion point")
        return
    
    print(f"Found insertion point at cell {insert_index}")
    print(f"Total cells before insertion: {len(nb['cells'])}")
    
    # Create the Double DQN cells
    cells_to_insert = []

    
    # Markdown cell 1: Introduction to Double DQN
    markdown1 = """#### Double DQN: Addressing Overestimation Bias

**The Problem with Standard DQN**

Standard DQN has a subtle but important flaw: it tends to **overestimate** Q-values. This happens because of how the max operator is used in the Q-learning update:

$
Q(s,a) \\leftarrow Q(s,a) + \\alpha \\left[ r + \\gamma \\max_{a'} Q(s', a') - Q(s,a) \\right]
$

**Why Overestimation Occurs:**

The same network is used for both:
1. **Selecting** the best action: $\\arg\\max_{a'} Q(s', a')$
2. **Evaluating** that action: $Q(s', a')$

This creates a **maximization bias**: if the Q-values have any estimation errors (which they always do), the max operation will tend to select actions with positive errors, leading to systematic overestimation.

**Example of the Problem:**

Imagine you're estimating the value of 3 actions, and your estimates have random errors:
- True values: [1.0, 1.0, 1.0] (all equal)
- Noisy estimates: [0.9, 1.2, 0.8] (with random errors)
- Standard DQN picks: max([0.9, 1.2, 0.8]) = 1.2
- This overestimates the true value of 1.0!

Over many updates, these overestimations accumulate and can hurt performance.

**The Double DQN Solution**

Double DQN (DDQN) addresses this by **decoupling action selection from action evaluation**:

1. Use the **online network** to select the best action
2. Use the **target network** to evaluate that action

**Double DQN Update Rule:**

$
Q(s,a) \\leftarrow Q(s,a) + \\alpha \\left[ r + \\gamma Q_{\\theta^-}\\left(s', \\arg\\max_{a'} Q_\\theta(s', a')\\right) - Q(s,a) \\right]
$

where:
- $Q_\\theta$ is the online network (selects action)
- $Q_{\\theta^-}$ is the target network (evaluates action)

**Key Insight:**

By using different networks for selection and evaluation, we reduce the correlation between the errors, which reduces overestimation bias.

**Benefits of Double DQN:**

- More accurate Q-value estimates
- Better performance on many tasks
- More stable learning
- Minimal computational overhead (we already have both networks!)

Let's implement Double DQN and compare it with standard DQN!"""
    
    cells_to_insert.append(create_markdown_cell(markdown1))

    
    # Code cell 1: Double DQN Agent Implementation
    code1 = """class DoubleDQNAgent(DQNAgent):
    \"\"\"Double DQN agent that reduces overestimation bias.
    
    Inherits from DQNAgent and only modifies the update method to use
    Double Q-learning: online network selects actions, target network evaluates them.
    \"\"\"
    
    def update(self):
        \"\"\"Update the agent using Double Q-learning.
        
        Key difference from standard DQN:
        - Online network selects the best action
        - Target network evaluates that action
        \"\"\"
        # Need enough samples in buffer
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample mini-batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Compute current Q-values
        current_q_values = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using Double Q-learning
        with torch.no_grad():
            # DOUBLE DQN: Use online network to SELECT actions
            next_actions = self.online_network(next_states).argmax(1)
            
            # DOUBLE DQN: Use target network to EVALUATE those actions
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Compute targets
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize the online network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()


print("Double DQN Agent implemented!")
print("\\nKey difference from standard DQN:")
print("  ✓ Online network SELECTS the best action")
print("  ✓ Target network EVALUATES that action")
print("  ✓ This decoupling reduces overestimation bias")"""
    
    cells_to_insert.append(create_code_cell(code1))

    
    # Markdown cell 2: Comparison setup
    markdown2 = """#### Comparing Standard DQN vs Double DQN

Now let's train both algorithms on the same environment and compare their performance. We'll look at:

1. **Learning curves**: How quickly do they learn?
2. **Final performance**: Which achieves better results?
3. **Stability**: Which is more consistent?
4. **Q-value estimates**: Do we see evidence of overestimation?

Let's run the comparison:"""
    
    cells_to_insert.append(create_markdown_cell(markdown2))

    
    # Code cell 2: Training comparison
    code2 = """def train_agent_comparison(agent_class, agent_name, env_name='CartPole-v1', 
                            num_episodes=300, max_steps=500, seed=42):
    \"\"\"
    Train an agent and return metrics for comparison.
    
    Args:
        agent_class: DQNAgent or DoubleDQNAgent class
        agent_name: Name for logging
        env_name: Gym environment name
        num_episodes: Number of training episodes
        max_steps: Max steps per episode
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with training metrics
    \"\"\"
    # Set seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    env = gym.make(env_name)
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"\\nTraining {agent_name} on {env_name}")
    print("=" * 60)
    
    # Create agent
    agent = agent_class(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    q_values = []  # Track Q-values to detect overestimation
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        episode_q = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state, training=True)
            
            # Track Q-values
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_vals = agent.online_network(state_tensor).max().item()
                episode_q.append(q_vals)
            
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            loss = agent.update()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        if episode_q:
            q_values.append(np.mean(episode_q))
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    return {
        'name': agent_name,
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'losses': losses,
        'q_values': q_values,
        'agent': agent
    }


# Train both agents with the same seed for fair comparison
print("Starting comparison experiment...")
print("This will train both DQN and Double DQN on CartPole-v1")
print("=" * 60)

# Train standard DQN
dqn_results = train_agent_comparison(
    DQNAgent, 
    "Standard DQN",
    num_episodes=300,
    seed=42
)

# Train Double DQN
ddqn_results = train_agent_comparison(
    DoubleDQNAgent,
    "Double DQN", 
    num_episodes=300,
    seed=42
)

print("\\n" + "=" * 60)
print("Training completed for both agents!")
print("=" * 60)"""
    
    cells_to_insert.append(create_code_cell(code2))

    
    # Markdown cell 3: Visualization intro
    markdown3 = """#### Visualizing the Comparison

Let's create comprehensive visualizations to compare the two algorithms:"""
    
    cells_to_insert.append(create_markdown_cell(markdown3))

    
    # Code cell 3: Comprehensive visualization
    code3 = """# Create comprehensive comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Prepare data
window = 20  # Moving average window

# Plot 1: Episode Rewards Comparison
ax1 = axes[0, 0]
ax1.plot(dqn_results['rewards'], alpha=0.2, color='blue', linewidth=0.5)
ax1.plot(ddqn_results['rewards'], alpha=0.2, color='red', linewidth=0.5)

# Moving averages
if len(dqn_results['rewards']) >= window:
    dqn_ma = np.convolve(dqn_results['rewards'], np.ones(window)/window, mode='valid')
    ax1.plot(range(window-1, len(dqn_results['rewards'])), dqn_ma, 
             color='blue', linewidth=2.5, label='Standard DQN')

if len(ddqn_results['rewards']) >= window:
    ddqn_ma = np.convolve(ddqn_results['rewards'], np.ones(window)/window, mode='valid')
    ax1.plot(range(window-1, len(ddqn_results['rewards'])), ddqn_ma, 
             color='red', linewidth=2.5, label='Double DQN')

ax1.axhline(y=195, color='green', linestyle='--', linewidth=2, 
            label='Solved Threshold', alpha=0.7)
ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Total Reward', fontsize=12)
ax1.set_title('Learning Curves: DQN vs Double DQN', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Q-Value Estimates Over Time
ax2 = axes[0, 1]
if dqn_results['q_values'] and ddqn_results['q_values']:
    ax2.plot(dqn_results['q_values'], alpha=0.3, color='blue', linewidth=0.5)
    ax2.plot(ddqn_results['q_values'], alpha=0.3, color='red', linewidth=0.5)
    
    # Moving averages
    if len(dqn_results['q_values']) >= window:
        dqn_q_ma = np.convolve(dqn_results['q_values'], np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(dqn_results['q_values'])), dqn_q_ma, 
                 color='blue', linewidth=2.5, label='Standard DQN')
    
    if len(ddqn_results['q_values']) >= window:
        ddqn_q_ma = np.convolve(ddqn_results['q_values'], np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(ddqn_results['q_values'])), ddqn_q_ma, 
                 color='red', linewidth=2.5, label='Double DQN')
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Average Max Q-Value', fontsize=12)
    ax2.set_title('Q-Value Estimates: Evidence of Overestimation?', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

# Plot 3: Training Loss Comparison
ax3 = axes[1, 0]
if dqn_results['losses'] and ddqn_results['losses']:
    ax3.plot(dqn_results['losses'], alpha=0.3, color='blue', linewidth=0.5)
    ax3.plot(ddqn_results['losses'], alpha=0.3, color='red', linewidth=0.5)
    
    # Moving averages
    if len(dqn_results['losses']) >= window:
        dqn_loss_ma = np.convolve(dqn_results['losses'], np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(dqn_results['losses'])), dqn_loss_ma, 
                 color='blue', linewidth=2.5, label='Standard DQN')
    
    if len(ddqn_results['losses']) >= window:
        ddqn_loss_ma = np.convolve(ddqn_results['losses'], np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(ddqn_results['losses'])), ddqn_loss_ma, 
                 color='red', linewidth=2.5, label='Double DQN')
    
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

# Plot 4: Performance Distribution
ax4 = axes[1, 1]
ax4.hist(dqn_results['rewards'], bins=30, alpha=0.5, color='blue', 
         label='Standard DQN', edgecolor='black')
ax4.hist(ddqn_results['rewards'], bins=30, alpha=0.5, color='red', 
         label='Double DQN', edgecolor='black')
ax4.axvline(x=np.mean(dqn_results['rewards']), color='blue', 
            linestyle='--', linewidth=2, alpha=0.7)
ax4.axvline(x=np.mean(ddqn_results['rewards']), color='red', 
            linestyle='--', linewidth=2, alpha=0.7)
ax4.set_xlabel('Total Reward', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('Reward Distribution Comparison', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Print detailed comparison statistics
print("\\n" + "=" * 70)
print("DETAILED COMPARISON: Standard DQN vs Double DQN")
print("=" * 70)

print("\\n📊 REWARD STATISTICS:")
print("-" * 70)
print(f"{'Metric':<30} {'Standard DQN':>18} {'Double DQN':>18}")
print("-" * 70)
print(f"{'Mean Reward':<30} {np.mean(dqn_results['rewards']):>18.2f} "
      f"{np.mean(ddqn_results['rewards']):>18.2f}")
print(f"{'Std Reward':<30} {np.std(dqn_results['rewards']):>18.2f} "
      f"{np.std(ddqn_results['rewards']):>18.2f}")
print(f"{'Max Reward':<30} {np.max(dqn_results['rewards']):>18.2f} "
      f"{np.max(ddqn_results['rewards']):>18.2f}")
print(f"{'Last 50 Episodes Mean':<30} {np.mean(dqn_results['rewards'][-50:]):>18.2f} "
      f"{np.mean(ddqn_results['rewards'][-50:]):>18.2f}")

print("\\n📈 Q-VALUE STATISTICS (Overestimation Check):")
print("-" * 70)
if dqn_results['q_values'] and ddqn_results['q_values']:
    print(f"{'Mean Q-Value':<30} {np.mean(dqn_results['q_values']):>18.2f} "
          f"{np.mean(ddqn_results['q_values']):>18.2f}")
    print(f"{'Max Q-Value':<30} {np.max(dqn_results['q_values']):>18.2f} "
          f"{np.max(ddqn_results['q_values']):>18.2f}")
    print(f"{'Final 50 Episodes Mean Q':<30} {np.mean(dqn_results['q_values'][-50:]):>18.2f} "
          f"{np.mean(ddqn_results['q_values'][-50:]):>18.2f}")

print("\\n🎯 CONVERGENCE:")
print("-" * 70)
# Find first episode where moving average exceeds 195
dqn_solved = -1
ddqn_solved = -1
threshold = 195
window_size = 100

for i in range(window_size, len(dqn_results['rewards'])):
    if np.mean(dqn_results['rewards'][i-window_size:i]) >= threshold:
        dqn_solved = i
        break

for i in range(window_size, len(ddqn_results['rewards'])):
    if np.mean(ddqn_results['rewards'][i-window_size:i]) >= threshold:
        ddqn_solved = i
        break

if dqn_solved > 0:
    print(f"{'Standard DQN solved at':<30} Episode {dqn_solved}")
else:
    print(f"{'Standard DQN':<30} Not solved")

if ddqn_solved > 0:
    print(f"{'Double DQN solved at':<30} Episode {ddqn_solved}")
else:
    print(f"{'Double DQN':<30} Not solved")

print("\\n" + "=" * 70)

# Determine winner
dqn_final = np.mean(dqn_results['rewards'][-50:])
ddqn_final = np.mean(ddqn_results['rewards'][-50:])

if ddqn_final > dqn_final:
    improvement = ((ddqn_final - dqn_final) / dqn_final) * 100
    print(f"\\n🏆 WINNER: Double DQN")
    print(f"   Improvement: {improvement:.1f}% better final performance")
elif dqn_final > ddqn_final:
    improvement = ((dqn_final - ddqn_final) / ddqn_final) * 100
    print(f"\\n🏆 WINNER: Standard DQN")
    print(f"   Improvement: {improvement:.1f}% better final performance")
else:
    print(f"\\n🤝 TIE: Both algorithms performed similarly")

print("=" * 70)"""
    
    cells_to_insert.append(create_code_cell(code3))

    
    # Markdown cell 4: Analysis and conclusions
    markdown4 = """#### Analysis: Why Double DQN Works Better

**Key Observations from the Comparison:**

1. **Q-Value Estimates**:
   - Standard DQN typically shows higher Q-values, indicating overestimation
   - Double DQN produces more conservative, accurate Q-value estimates
   - This is evidence of the overestimation bias being reduced

2. **Learning Stability**:
   - Double DQN often shows smoother learning curves
   - Less variance in performance across episodes
   - More consistent convergence to good policies

3. **Final Performance**:
   - Double DQN frequently achieves better or equal final performance
   - The improvement is more pronounced in complex environments
   - CartPole is relatively simple, so differences may be subtle

4. **Computational Cost**:
   - Double DQN has virtually no additional computational cost
   - Same network architecture and training time
   - Only the update rule changes slightly

**When Does Double DQN Help Most?**

Double DQN provides the biggest benefits when:
- The environment has stochastic rewards or transitions
- The action space is large
- Q-value estimation is noisy
- Long-term planning is important

**Practical Recommendations:**

✅ **Use Double DQN** as your default choice - it's strictly better than standard DQN with no downsides

✅ **Combine with other improvements** like:
   - Prioritized Experience Replay
   - Dueling Networks
   - Noisy Networks for exploration

✅ **Monitor Q-values** during training to detect overestimation issues

**Mathematical Insight:**

The key insight is that by decoupling action selection from evaluation, we reduce the positive bias:

$
\\mathbb{E}[\\max_a Q(s,a)] \\geq \\max_a \\mathbb{E}[Q(s,a)]
$

This inequality (Jensen's inequality for the max function) shows that taking the max of noisy estimates gives a biased result. Double DQN mitigates this by using independent estimates.

**Next Steps:**

Double DQN is a foundational improvement to DQN. Modern deep RL often combines it with other techniques like:
- **Dueling DQN**: Separate value and advantage streams
- **Prioritized Replay**: Sample important transitions more frequently  
- **Rainbow DQN**: Combines multiple improvements into one algorithm

In the next sections, we'll explore policy gradient methods, which take a fundamentally different approach to RL!"""
    
    cells_to_insert.append(create_markdown_cell(markdown4))
    
    # Insert all cells into the notebook
    # Insert after the cell that mentions Double DQN
    insert_position = insert_index + 1
    
    for i, cell in enumerate(cells_to_insert):
        nb['cells'].insert(insert_position + i, cell)
    
    print(f"Inserted {len(cells_to_insert)} cells at position {insert_position}")
    print(f"Total cells after insertion: {len(nb['cells'])}")
    
    # Save the updated notebook
    save_notebook('reinforcement_learning_zero_to_hero.ipynb', nb)
    print("\\nNotebook saved successfully!")
    print("Double DQN content has been added to the notebook.")


if __name__ == "__main__":
    main()
