# main.py
import gymnasium as gym
import tensorflow as tf
from keras import layers
import numpy as np
import random
from collections import deque
import time
import ale_py

# I. Pengaturan Lingkungan
# ==============================================================================
# Kita akan menggunakan pustaka Gymnasium, penerus OpenAI's Gym, untuk membuat
# lingkungan Ms. Pac-Man. Arcade Learning Environment (ALE) menyediakan
# lingkungan permainan. Kita juga akan menggunakan beberapa "wrapper" untuk
# memproses awal bingkai permainan, membuatnya lebih cocok untuk melatih
# jaringan saraf.


def make_env(env_name, seed=None):
    """
    Membuat dan memproses awal lingkungan.
    """
    if seed is not None:
        gym.utils.seeding.np_random(seed)

    # Create the base environment
    # env = gym.make(env_name, render_mode="human")
    env = gym.make(env_name)

    # Apply wrappers for preprocessing
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)

    return env


# II. Model Deep Q-Network (DQN)
# ==============================================================================
# Agen kita akan menggunakan Deep Q-Network (DQN) untuk mempelajari kebijakan
# yang optimal. DQN adalah jaringan saraf tiruan konvolusional (CNN) yang
# mengambil bingkai permainan yang sudah diproses sebagai masukan dan
# menghasilkan nilai Q untuk setiap tindakan yang mungkin.


def create_dqn_model(input_shape, num_actions):
    """
    Membuat model DQN.
    """
    return tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Permute((2, 3, 1)),  # Change from (frames, H, W) to (H, W, frames)
            layers.Conv2D(
                32,
                (8, 8),
                strides=(4, 4),
                activation="relu",
                data_format="channels_last",
            ),
            layers.Conv2D(
                64,
                (4, 4),
                strides=(2, 2),
                activation="relu",
                data_format="channels_last",
            ),
            layers.Conv2D(
                64,
                (3, 3),
                strides=(1, 1),
                activation="relu",
                data_format="channels_last",
            ),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(num_actions, activation="linear"),
        ]
    )


# III. Buffer Pengalaman (Replay Buffer)
# ==============================================================================
# Untuk menstabilkan pelatihan, kita akan menggunakan buffer pengalaman untuk
# menyimpan pengalaman agen. Agen kemudian akan mengambil sampel dari buffer ini
# untuk melatih model DQN. Ini membantu memutus korelasi antara pengalaman
# berturut-turut dan meningkatkan stabilitas proses pelatihan.


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# IV. Agen DQN
# ==============================================================================
# Kelas DQNAgent merangkum perilaku agen, termasuk model DQN, buffer pengalaman,
# dan kebijakan epsilon-greedy untuk pemilihan tindakan.


class DQNAgent:
    def __init__(
        self,
        env,
        model,
        replay_buffer,
        learning_rate,
        gamma,
        epsilon,
        epsilon_decay,
        min_epsilon,
    ):
        self.env = env
        self.model = model
        self.replay_buffer = replay_buffer
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = tf.keras.losses.Huber()

        # Target network
        self.target_model = tf.keras.models.clone_model(model)
        self.target_model.set_weights(model.get_weights())
        self.target_update_frequency = 1000  # Update target model every 1000 steps
        self.train_step_counter = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            # Use the main model for action selection
            q_values = self.model(np.expand_dims(state, axis=0))
            return np.argmax(q_values[0])

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        samples = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)

        # Use the target model to predict future rewards
        future_rewards = self.target_model(next_states)
        target_q_values = rewards + self.gamma * tf.reduce_max(
            future_rewards, axis=1
        ) * (1 - dones)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            action_masks = tf.one_hot(actions, self.env.action_space.n)
            predicted_q_values = tf.reduce_sum(action_masks * q_values, axis=1)
            loss = self.loss_fn(target_q_values, predicted_q_values)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # Update target model periodically
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_frequency == 0:
            self.target_model.set_weights(self.model.get_weights())

    def save_model(self, path):
        """
        Menyimpan model DQN utama ke jalur yang ditentukan.
        """
        self.model.save(path)


# V. Loop Pelatihan
# ==============================================================================
# Loop pelatihan adalah tempat agen berinteraksi dengan lingkungan, menyimpan
# pengalaman, dan melatih model DQN.


def main():
    # Hyperparameters
    env_name = "ALE/MsPacman-v5"
    learning_rate = 0.00025
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    replay_buffer_capacity = 100000
    batch_size = 32
    num_episodes = 10  # Reduced for smoother demonstration

    # Initialization
    env = make_env(env_name)
    model = create_dqn_model(env.observation_space.shape, env.action_space.n)
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    agent = DQNAgent(
        env,
        model,
        replay_buffer,
        learning_rate,
        gamma,
        epsilon,
        epsilon_decay,
        min_epsilon,
    )

    # Training
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            agent.replay_buffer.store(state, action, reward, next_state, done)
            agent.train(batch_size)

            state = next_state

        print(
            f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}"
        )

    # Save the trained model
    model_save_path = "dqn_model.keras"
    agent.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")

    env.close()


if __name__ == "__main__":
    main()
