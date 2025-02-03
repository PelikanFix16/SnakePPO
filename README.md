# 🐍 Snake PPO - Reinforcement Learning with Proximal Policy Optimization (PPO)

## 📌 Overview
This repository contains an implementation of **Proximal Policy Optimization (PPO)** for training an AI agent to play the **Snake** game using **Deep Reinforcement Learning**. The agent learns to navigate and maximize its score by consuming food while avoiding collisions with the walls and itself.

## 📁 Project Structure
```
📦 SnakePPO-v2
 ┣ 📜 main.py          # Main training script using PPO
 ┣ 📜 train.py         # Training loop and model saving
 ┣ 📜 test.py          # Testing script for evaluating the trained model
 ┣ 📜 memory.py        # Replay memory implementation
 ┣ 📜 network.py       # Neural network architecture
 ┣ 📜 ppo_snake_model.pth # Trained model weights
 ┣ 📜 README.md        # Project documentation
```

## 🚀 Installation
To run this project, first install the required dependencies:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
pip install gym-snake-game
pip install gym
pip install gymnasium
```

## 🎮 Running the Project
### 1️⃣ Train the Model
To train the Snake agent using PPO, run:
```bash
python main.py
```
This script initializes the environment, trains the PPO agent, and saves the model weights.

### 2️⃣ Test the Model
To evaluate the trained model, run:
```bash
python test.py
```
This script loads the saved model and runs a simulation of the Snake game.

## 🔬 PPO Algorithm Overview
Proximal Policy Optimization (PPO) is a policy gradient method that improves training stability and efficiency by introducing a clipped objective function. This prevents drastic updates that could negatively impact training. PPO is widely used in reinforcement learning applications due to its simplicity and performance.

### Key Features of PPO in this Project:
- **Actor-Critic architecture** for better stability
- **Advantage estimation** for more efficient training
- **Clipped objective function** to maintain robust policy updates

## 📜 To-Do List
- [ ] Implement reward shaping for better training performance
- [ ] Add visualization for training progress
- [ ] Experiment with different hyperparameters for optimization

## 👨‍💻 Author
Developed by **PelikanFix16**. Feel free to reach out or contribute to this project!

## ⭐ Contributions
Pull requests and suggestions are welcome! If you find this project useful, give it a **star** ⭐ on GitHub!

## ⚖️ License
This project is licensed under the **MIT License**.

