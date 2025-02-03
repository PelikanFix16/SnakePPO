import gym_snake_game
import numpy as np
import torch
from train import Train

# Opcje środowiska Snake
options = {
    "fps": 35,
    "max_step": 5000,
    "init_length": 4,
    "food_reward": 100.0,
    "dist_reward": None,
    "living_bonus": -0.01,
    "death_penalty": -100.0,
    "width": 40,
    "height": 40,
    "block_size": 20,
    "background_color": (255, 169, 89),
    "food_color": (255, 90, 90),
    "head_color": (197, 90, 255),
    "body_color": (89, 172, 255),
}

# Inicjalizacja środowiska
env = gym_snake_game.make("Snake-v0", render_mode="human", **options)
obs = env.reset()[0]

# Rozmiar wejściowy i wyjściowy
input_size = len(obs)
output_size = env.action_space.n

# Inicjalizacja agenta PPO
agent = Train(input_size, output_size)

# Parametry pętli uczącej
num_episodes = 100000
print_every = 100
save_every = 2000
model_save_path = "ppo_snake_model.pth"
agent.network.load_state_dict(torch.load(model_save_path))
agent.network.eval()

rewards = 0
# Główna pętla ucząca
for episode in range(1, num_episodes + 1):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        # Wybór akcji
        action, log_prob = agent.action(state, train=False)
        value = agent.value(state)

        # Wykonanie akcji
        next_state, reward, done, truncated, info = env.step(action)
        reward = reward / 100.0  # Normalizacja nagrody
        rewards += reward

        # Przejście do następnego stanu
        state = next_state
        total_reward += reward

        if done or truncated:
            break
    print(f"Episode: {episode}, Reward: {rewards}")
    rewards = 0


# Zamknięcie środowiska
env.close()
