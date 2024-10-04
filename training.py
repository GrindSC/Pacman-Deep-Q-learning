from agent import *
from environment import *

# Setup env
env = Pacman(board)
env.reset()

# Setup model
state_size = env.get_number_of_states()
action_size = 4
learning_rate = 0.001

model = Sequential()
model.add(Dense(state_size, input_dim=state_size, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(action_size))  # wyjście
model.compile(loss="mean_squared_error",
              optimizer=Adam(learning_rate=learning_rate))

# Setup agent
agent = DQNAgent(action_size, learning_rate, model, get_legal_actions=env.get_possible_actions)

# Initialize parameters
done = False
batch_size = 64  # Size of the batch for training
EPISODES = 1000  # Total number of training episodes
env.turn_off_display()  # Turn off environment display for training efficiency

for episode in range(EPISODES):
    start_time = tm.time()  # Performance measurement
    episode_rewards = []  # Training history

    # Run the environment for a fixed number of iterations
    for _ in range(10):
        total_reward = 0
        env_state = env.reset()
        state = env_state

        for time_step in range(500):  # Limit the number of steps per episode
            action = agent.get_action(state)
            next_state_env, reward, done, _ = env.step(action)
            total_reward += reward

            # Store experience in memory for later replay
            agent.remember(
                np.array([to_categorical(state, num_classes=state_size)]),
                action,
                reward,
                np.array([to_categorical(next_state_env, num_classes=state_size)]),
                done
            )
            state = next_state_env

            if done:
                break

        # Train the agent using a batch of experiences if there is enough memory
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        episode_rewards.append(total_reward)

    end_time = tm.time()
    print(
        "Episode #{}\tMean Reward = {:.3f}\tEpsilon = {:.3f}\tTime = {:.2f}s".format(
            episode, np.mean(episode_rewards), agent.epsilon, end_time - start_time
        )
    )

    # Save the model + early stopping if the performance exceeds a certain threshold
    if np.mean(total_reward) > 450:
        model.save('model.h5')
        break