import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

MAX_STEPS = 100
NUMBER_OF_EPISODES = 6000

class Buffer():
    def __init__(self, buffer_size, num_of_states):
        self.buffer_size_ = buffer_size
        self.buffer_count_ = 0
        self.buffer_full_ = False
        self.buffer_ind_ = 0

        self.state_buffer_ = np.zeros((buffer_size, num_of_states))
        self.reward_buffer_ = np.zeros((buffer_size, 1))
        self.action_buffer_ = np.zeros((buffer_size, 1))
        self.done_buffer_ = np.zeros((buffer_size, 1))
        self.new_state_buffer_ = np.zeros((buffer_size, num_states))

    def get_buffer_ind(self):
        return self.buffer_ind_

    def set_buffer_ind(self, i):
        self.buffer_ind_ = i

    def get_buffer_count(self):
        return self.buffer_count_

    def set_buffer_count(self, value):
        self.buffer_count_ = value

    def get_buffer_size(self):
        return self.buffer_size_

    def get_buffer_full(self):
        return self.buffer_full_

    def set_buffer_full(self, bool):
        self.buffer_full_ = bool

    def update_buffer_tables(self, state, action, reward, new_state, done):
        i = self.get_buffer_ind()
        self.state_buffer_[i, :] = state
        self.action_buffer_[i] = action
        self.reward_buffer_[i] = reward
        self.new_state_buffer_[i, :] = new_state
        self.done_buffer_[i] = done

def create_NnetTaxi(num_of_states=500):
    input_layer = tf.keras.layers.Input(shape=(num_of_states,), name="input_layer")
    dense_1 = tf.keras.layers.Dense(40, activation="relu", name="dense_1")(input_layer)
    dense_2 = tf.keras.layers.Dense(32, activation="relu", name="dense_2")(dense_1)
    output_layer = tf.keras.layers.Dense(6, activation="linear", name="output_layer")(dense_2)
    mlp_model = tf.keras.Model(inputs=input_layer,
                                 outputs=output_layer,
                                 name="Nnet_taxi_driver")
    return mlp_model

def evaluate_policy(q_table, max_steps=MAX_STEPS):
    episodes_test = 1000
    episode_rewards = []

    for episode in range(episodes_test):
        state, info = env_sim.reset()
        step = 0
        done = False
        total_rewards = 0

        while not done and (step < max_steps):
            action = np.argmax(q_table[state, :])
            new_state, reward, done, _, info = env_sim.step(action)
            total_rewards += reward
            state = new_state
            step += 1

        episode_rewards.append(total_rewards)
    env_sim.close()
    avg_reward = sum(episode_rewards)/episodes_test

    return_stats = {"average": avg_reward,
                    "mean": np.mean(episode_rewards),
                    "min": np.min(episode_rewards),
                    "max": np.max(episode_rewards),
                    "std": np.std(episode_rewards)}

    return return_stats

def simulate_best_policy(env, Q, max_steps = MAX_STEPS):
    state, info_state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done and (steps < max_steps):
        action = np.argmax(Q[state, :])
        state, reward, done, _, info_state = env.step(action)
        env.render()
        total_reward += reward
        steps += 1
    print("Total reward:", total_reward)
    env.close()

# Main
env_sim = gym.make('Taxi-v3')
env_human = gym.make('Taxi-v3', render_mode="human")

num_states = env_sim.observation_space.n
num_actions = env_sim.action_space.n
print("# of states: ", num_states)
print("# of actions: ", num_actions)

q_table = np.zeros((num_states, num_actions))
print("Q table: ", q_table.shape)

model = create_NnetTaxi()
model.summary()

# Parameters
alpha = 0.1 #learning rate
gamma = 0.9 #discount factor
epsilon = 1 #exploration rate
decay_rate = 0.997 #rate for the decreasing exploration
min_epsilon = 0.01 #min value for exploring

n_episodes = NUMBER_OF_EPISODES #number of simulation episodes
max_steps = MAX_STEPS #number of maximum actions in one simulation
history = [] #evaluation history

buffer = Buffer()
tr_batch_size = 100
tf_freq = 10

for episode in range(n_episodes):
    state, info_state = env_sim.reset()
    one_hot_state = tf.one_hot(state, num_states)
    one_hot_state = tf.reshape(one_hot_state, [1, num_states])
    done = False
    steps = 0

    while not done and (steps < max_steps):
        print(one_hot_state.shape)
        # Based on epsilon-greedy policy explore sometimes
        if np.random.uniform() > epsilon: # Can try np.random.random() also
            q_net_pred = model.predict(one_hot_state)
            action = np.argmax(q_net_pred)
        else:
            action = np.random.randint(0, env_sim.action_space.n)

        new_state, reward, done, _, info_state = env_sim.step(action)
        # q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * \
        #                         (reward + gamma * (np.max(q_table[new_state, :])))

        if buffer.get_buffer_count() < buffer.get_buffer_size():
            count = buffer.get_buffer_count()
            buffer.set_buffer_ind(count)
        else:
            buffer.set_buffer_count(0)
            count = buffer.get_buffer_count()
            buffer.set_buffer_ind(count)
            buffer.set_buffer_full(True)

        buffer.update_buffer_tables(state, action, reward, new_state, done)

        state = new_state
        steps += 1
        buffer.set_buffer_count(buffer.get_buffer_count() + 1)

    if episode % (NUMBER_OF_EPISODES/5) == 0 or\
            episode ==1 or\
            episode == NUMBER_OF_EPISODES - 1:

        int_policy = np.argmax(q_table, axis=1)
        int_stats = evaluate_policy(q_table)
        history.append([episode,
                        int_stats["mean"],
                        int_stats["min"],
                        int_stats["max"],
                        int_stats["std"]])

    epsilon = epsilon * decay_rate
    epsilon = max(epsilon, min_epsilon)
    #print("Epsilon: ", epsilon, " Episode: ", episode)
    #print("Done: ", done," With: ", steps, "steps")

    # Next train the network!!

# Includes "average", "mean", "min", "max", "std"
print("Policy statistics: ", evaluate_policy(q_table))

# Plot histogram of training episode stats
history = np.array(history)
plt.plot(history[:, 0], history[:, 1])
plt.fill_between(history[:, 0], history[:, 1]-history[:, 4], history[:, 1]+history[:, 4],
                 alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99', linewidth=0)
plt.show()

# Watch the best policy
#simulate_best_policy(env_human, q_table, 20)

env_sim.close()
env_human.close()
