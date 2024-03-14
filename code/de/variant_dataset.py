from torch.utils.data import Dataset

class VariantDataset(Dataset):
    def __init__(self, sequences, fitness):
        self.sequences = sequences
        self.fitness = fitness

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        fit = self.fitness[idx]
        return seq, fit
