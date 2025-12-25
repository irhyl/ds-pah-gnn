import os
import glob
import pickle
from torch_geometric.data import Dataset
from converter import raw_to_pyg

class MicrogridDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir: Path to the folder containing 'shard_*.pkl' files.
                      (e.g., '../microgrid-design/dataset')
        """
        self.root_dir = root_dir
        self.shard_paths = sorted(glob.glob(os.path.join(root_dir, "shard_*.pkl")))
        
        if not self.shard_paths:
            raise FileNotFoundError(f"No .pkl shards found in {root_dir}. Please run generate.py first.")
            
        # Load first shard to determine chunk size (assuming constant size)
        with open(self.shard_paths[0], "rb") as f:
            first_chunk = pickle.load(f)
            self.chunk_size = len(first_chunk)
            
        # Calculate exact total length by checking the last shard
        # (The last shard might be smaller than chunk_size)
        with open(self.shard_paths[-1], "rb") as f:
            last_chunk = pickle.load(f)
            
        self.total_len = (len(self.shard_paths) - 1) * self.chunk_size + len(last_chunk)
        
        # Cache for performance
        self._cached_shard_idx = -1
        self._cached_data = None
        
        super().__init__(root=None, transform=None, pre_transform=None)

    def len(self):
        return self.total_len

    def get(self, idx):
        """
        Maps global index to (shard_file, local_index).
        """
        shard_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size
        
        # Load shard if not in memory
        if shard_idx != self._cached_shard_idx:
            # print(f"Loading shard {shard_idx}...") # Debug
            with open(self.shard_paths[shard_idx], "rb") as f:
                self._cached_data = pickle.load(f)
            self._cached_shard_idx = shard_idx
            
        raw_snapshot = self._cached_data[local_idx]
        
        # Convert to PyG Data on the fly
        data = raw_to_pyg(raw_snapshot)
        return data