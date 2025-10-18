# SARSA Learning Algorithm


## AIM
To implement SARSA Learning Algorithm.

## PROBLEM STATEMENT
The problem might involve teaching an agent to interact optimally with an environment (e.g., gym-walk), where the agent must learn to choose actions that maximize cumulative rewards using RL algorithms like SARSA and Value Iteration.



## SARSA LEARNING ALGORITHM
Initialize the Q-table, learning rate α, discount factor γ, exploration rate ϵ, and the number of episodes.
For each episode, start in an initial state s, and choose an action a using the ε-greedy policy.
Take action a, observe the reward r and the next state s′ , and choose the next action a′ using the ε-greedy policy.
Update the Q-value for the state-action pair (s,a) using the SARSA update rule.
Update the current state to s′ and the current action to a′.
Repeat steps 3-5 until the episode reaches a terminal state.
After each episode, decay the exploration rate 𝜖 and learning rate α, if using decay schedules.
Return the Q-table and the learned policy after completing all episodes.

## SARSA LEARNING FUNCTION
### Name:ALAN ZION H
### Register Number:212223240004

def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilon = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
      state, done = env.reset(), False
      action = select_action(state, Q, epsilon[e])
      while not done:
        next_state, reward, done, _ = env.step(action)
        next_action = select_action(next_state, Q, epsilon[e])
        td_target = reward + gamma * Q[next_state][next_action] * (not done)
        td_error = td_target - Q[state][action]
        Q[state][action] = Q[state][action] + alphas[e] * td_error
        state, action = next_state, next_action
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
    
## OUTPUT:
<img width="699" height="605" alt="image" src="https://github.com/user-attachments/assets/c8d90614-8785-44b5-b86f-e02eb203635b" />

<img width="367" height="68" alt="image" src="https://github.com/user-attachments/assets/e9f69190-faa3-4bba-bf41-e5a78cb80466" />

<img width="1518" height="587" alt="image" src="https://github.com/user-attachments/assets/0dfbfaec-1865-45cf-9b07-371b2f419944" />

<img width="1513" height="581" alt="image" src="https://github.com/user-attachments/assets/d4406bd1-cf64-4650-9a2a-f6865b936add" />


## RESULT:

Thus, to implement SARSA learning algorithm is executed successfully.
