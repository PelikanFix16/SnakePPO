from memory import PPOBuffer
from network import ActorCritic
import torch
import numpy as np


class Train:
    def __init__(self, state_dim, action_dim, gamma=0.95, clip_eps=0.1, num_epochs=12):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.buffer = PPOBuffer(3000, state_dim, device=self.device)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.num_epochs = num_epochs
        self.critic_optim = torch.optim.Adam(self.network.critic.parameters(), lr=1e-3)
        self.actor_optim = torch.optim.Adam(self.network.actor.parameters(), lr=1e-3)
        self.stepLrActor = torch.optim.lr_scheduler.StepLR(
            self.actor_optim, step_size=10000, gamma=0.5
        )
        self.stepLrCritic = torch.optim.lr_scheduler.StepLR(
            self.critic_optim, step_size=10000, gamma=0.5
        )

    def action(self, state, train=True):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        mean = self.network.actor(state)
        dist = torch.distributions.Categorical(mean)
        if train:
            action = dist.sample()
        else:
            action = torch.argmax(mean)
        log_prob = dist.log_prob(action)
        return action.cpu().detach().numpy(), log_prob.cpu().detach().numpy()

    def value(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        return self.network.critic(state).squeeze().cpu().detach().numpy()

    def compute_returns_and_advantages(
        self, rewards, dones, values, gamma=0.99, lam=0.94
    ):
        # Zamień tensory na NumPy do przeliczeń
        rewards = rewards.cpu().numpy()
        dones = dones.cpu().numpy()
        values = values.cpu().numpy()

        # Oblicz returns
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Dopasuj wartości dla `deltas`
        next_values = np.roll(
            values, -1
        )  # Przesunięcie w lewo (ostatni element staje się zerem)
        next_values[-1] = 0  # Wypełnienie końcowej wartości zerem
        deltas = rewards + gamma * next_values * (1 - dones) - values

        # Oblicz advantages
        advantages = self.discount_cumsum(deltas, gamma * lam)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        return returns, advantages

    def discount_cumsum(self, x, discount):
        result = []
        cumsum = 0
        for value in reversed(x):
            cumsum = value + discount * cumsum
            result.insert(0, cumsum)
        return result

    def ppo_loss(self, log_probs, old_log_probs, advantages, epsilon=0.2):
        assert (
            log_probs.shape == old_log_probs.shape
        ), "Log_probs and old_log_probs mismatch"
        assert log_probs.shape == advantages.shape, "Log_probs and advantages mismatch"

        ratios = torch.exp(log_probs - old_log_probs)  # r_t(theta)
        clip_adv = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
        return -torch.min(ratios * advantages, clip_adv).mean()

    def update(self):
        # Pobierz dane z bufora
        states, actions, rewards, dones, old_log_probs, old_values = (
            self.buffer.get_all()
        )
        # Oblicz returns i advantages
        returns, advantages = self.compute_returns_and_advantages(
            rewards, dones, old_values, self.gamma
        )

        for _ in range(self.num_epochs):
            # Oblicz nową politykę
            new_means = self.network.actor(states)
            dist = torch.distributions.Categorical(new_means)
            new_log_probs = dist.log_prob(actions)

            # Oblicz stratę polityki
            policy_loss = self.ppo_loss(
                new_log_probs, old_log_probs, advantages, self.clip_eps
            )

            # Oblicz stratę krytyka
            values = self.network.critic(states).squeeze()
            critic_loss = ((returns - values) ** 2).mean()

            # Aktualizacja krytyka
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # Aktualizacja aktora
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()
            self.stepLrActor.step()
            self.stepLrCritic.step()

        # Wyczyść bufor po aktualizacji
        self.buffer.clear()
