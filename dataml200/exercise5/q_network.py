import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

MAX_STEPS = 100
NUMBER_OF_EPISODES = 100

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

    def get_state(self):
        return self.state_buffer_

    def get_reward(self):
        return self.reward_buffer_

    def get_action(self):
        return self.action_buffer_

    def get_done(self):
        return self.done_buffer_

    def get_new_state(self):
        return self.new_state_buffer_

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

def evaluate_policy(model, num_states, max_steps=10):
    print("evaluate")
    episodes_test = 10
    episode_rewards = []

    for episode in range(episodes_test):
        state, info = env_sim.reset()
        one_hot_state = tf.one_hot(state, num_states)
        one_hot_state = tf.reshape(one_hot_state, [1, num_states])
        step = 0
        done = False
        total_rewards = 0

        while not done and (step < max_steps):
            q_net_pred = model.predict(one_hot_state)
            action = np.argmax(q_net_pred)
            new_state, reward, done, _, info = env_sim.step(action)
            one_hot_new_state = tf.one_hot(new_state, num_states)
            one_hot_new_state = tf.reshape(one_hot_new_state, [1, num_states])
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

def simulate_best_policy(env, model, max_steps = MAX_STEPS):
    state, info_state = env.reset()
    one_hot_state = tf.one_hot(state, num_states)
    one_hot_state = tf.reshape(one_hot_state, [1, num_states])
    done = False
    total_reward = 0
    steps = 0
    while not done and (steps < max_steps):
        q_net_pred = model.predict(one_hot_state)
        action = np.argmax(q_net_pred)
        state, reward, done, _, info_state = env.step(action)
        one_hot_state = tf.one_hot(state, num_states)
        one_hot_state = tf.reshape(one_hot_state, [1, num_states])
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
loss_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['mse'])
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

buffer = Buffer(1000, num_states)
tr_batch_size = 100
tr_freq = 30

for episode in range(n_episodes):
    print(episode)
    state, info_state = env_sim.reset()
    one_hot_state = tf.one_hot(state, num_states)
    one_hot_state = tf.reshape(one_hot_state, [1, num_states])
    done = False
    steps = 0

    while not done and (steps < max_steps):
        # Based on epsilon-greedy policy explore sometimes
        if np.random.uniform() > epsilon: # Can try np.random.random() also
            print(one_hot_state.shape)
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
        int_stats = evaluate_policy(model, num_states)
        history.append([episode,
                        int_stats["mean"],
                        int_stats["min"],
                        int_stats["max"],
                        int_stats["std"]])

    epsilon = epsilon * decay_rate
    epsilon = max(epsilon, min_epsilon)
    #print("Epsilon: ", epsilon, " Episode: ", episode)
    #print("Done: ", done," With: ", steps, "steps")

    if buffer.get_buffer_full() and episode % tr_freq == 0:
        X = np.zeros((tr_batch_size, num_states))
        Y = np.zeros((tr_batch_size, num_actions))
        for ind, tr_ind in enumerate(np.random.randint(buffer.get_buffer_size(),
                                                       size=tr_batch_size)):
            buffer_state = buffer.get_state()
            #print(buffer_state.shape)
            #buffer_state = tf.reshape(buffer_state, [1, num_states])
            #print(buffer_state.shape)
            one_hot_buf_state = tf.one_hot(buffer_state, num_states)
            print(one_hot_buf_state.shape)
            #one_hot_buf_state = tf.reshape(one_hot_buf_state, [1, num_states])
            new_state_buffer = buffer.get_new_state()
            #new_state_buffer = tf.reshape(new_state_buffer, [1, num_states])
            #one_hot_buf_newstate = tf.one_hot(new_state_buffer, num_states)
            #one_hot_buf_newstate = tf.reshape(one_hot_buf_newstate, [1, num_states])
            action_buffer = buffer.get_action()
            done_buffer = buffer.get_done()
            reward_buffer = buffer.get_reward()

            print(X[ind, :].shape, " ", buffer_state[tr_ind, :].shape)
            buffer_state_reshaped = tf.reshape(buffer_state[tr_ind, :], [1, num_states])
            buffer_state_squeezed = tf.squeeze(buffer_state_reshaped)
            print(buffer_state_reshaped.shape)
            X[ind, :] = buffer_state_squeezed
            print(X[ind, :].shape, " ", buffer_state_reshaped.shape)
            Y[ind, :] = model.predict(buffer_state_reshaped)
            if done_buffer[tr_ind]:
                Y[ind, int(action_buffer[tr_ind])] = reward_buffer[tr_ind]
            else:
                buffer_new_state_reshaped = tf.reshape(new_state_buffer[tr_ind, :],
                                                       [1, num_states])
                Y[ind, int(action_buffer[tr_ind])] = \
                    reward_buffer[tr_ind] + gamma *\
                    np.max(model.predict(buffer_new_state_reshaped))
        model.fit(X, Y, epochs=5, verbose=1)

# Includes "average", "mean", "min", "max", "std"
print("Policy statistics: ", evaluate_policy(model, num_states))

# Plot histogram of training episode stats
history = np.array(history)
plt.plot(history[:, 0], history[:, 1])
plt.fill_between(history[:, 0], history[:, 1]-history[:, 4], history[:, 1]+history[:, 4],
                 alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99', linewidth=0)
plt.show()

# Watch the best policy
simulate_best_policy(env_human, model, 20)

env_sim.close()
env_human.close()
