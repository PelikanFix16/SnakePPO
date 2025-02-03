import torch


class PPOBuffer:
    def __init__(self, buffer_size, state_dim, device="cpu"):
        self.buffer_size = buffer_size
        self.ptr = 0  # wskaźnik pozycji do zapisu
        self.device = device

        # Bufory danych
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32)
        self.actions = torch.zeros(buffer_size, dtype=torch.int32)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32)
        self.values = torch.zeros(buffer_size, dtype=torch.float32)

    def store(self, state, action, reward, done, log_prob, value):
        """
        Zapisuje jedno doświadczenie w buforze.
        """
        if self.ptr >= self.buffer_size:
            return
        self.states[self.ptr] = torch.tensor(state, dtype=torch.float32)
        self.actions[self.ptr] = torch.tensor(action, dtype=torch.int32)
        self.rewards[self.ptr] = torch.tensor(reward, dtype=torch.float32)
        self.dones[self.ptr] = torch.tensor(done, dtype=torch.float32)
        self.log_probs[self.ptr] = torch.tensor(log_prob, dtype=torch.float32)
        self.values[self.ptr] = torch.tensor(value, dtype=torch.float32)
        self.ptr += 1

    def get_all(self):
        """
        Zwraca wszystkie dane w buforze na odpowiednie urządzenie.
        """
        return (
            self.states[: self.ptr].to(self.device),
            self.actions[: self.ptr].to(self.device),
            self.rewards[: self.ptr].to(self.device),
            self.dones[: self.ptr].to(self.device),
            self.log_probs[: self.ptr].to(self.device),
            self.values[: self.ptr].to(self.device),
        )

    def clear(self):
        """
        Czyści bufor.
        """
        self.ptr = 0
