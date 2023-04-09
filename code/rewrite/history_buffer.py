import torch
import random

class HistoryBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        
    def __len__(self):
        return len(self.buffer)

    def _make_space(self, max_del_size):
        current_available_space = self.max_size - len(self)
        del_size = max(0, max_del_size - current_available_space)

        if del_size == 0:
            return

        del_indexes = random.sample(range(0, len(self)), del_size)

        for del_idx in del_indexes:
            del self.buffer[del_idx]

    def add(self, batch):
        self._make_space(len(batch))

        for item in batch:
            self.buffer.append(item.detach())

    def sample_batch(self, batch_size):
        return torch.stack(random.sample(self.buffer, batch_size))

    def randomise_existing_batch(self, existing_batch):
        if len(self) < existing_batch.shape[0] / 2:
            return existing_batch
        
        new_batch = []

        for item in existing_batch:
            if random.uniform(0, 1) < 0.5:
                new_batch.append(item.detach())
            else:
                new_batch.append(self.buffer[random.randint(0, len(self) - 1)])

        return torch.stack(new_batch)