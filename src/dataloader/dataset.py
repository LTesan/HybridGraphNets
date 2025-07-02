import torch
from torch.utils.data import Dataset
import os

from src.utils.pp import find_nonzero_indices, label_all, create_wedge_index, set_u


class GraphDataset(Dataset):
    def __init__(self, args, data_dir, rollout = False):
        """
        Args:
            args: Arguments including parameters like att_radius.
            data_dir (str): Directory with all the .pt files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.args = args
        ratio = args.ratio
        self.rollout = rollout
        self.data_dir = data_dir
        self.att_radius = args.att_radius
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        # If ratio is specified, select a subset of files
        if ratio < 1.0:
            num_files = int(len(self.data_files) * ratio)
            self.data_files = self.data_files[:num_files]

        # Loads hole sims from .pt files
        if self.rollout:
            self.data = []
            for file in self.data_files:
                file_path = os.path.join(self.data_dir, file)
                simulations = torch.load(file_path, weights_only=False)
                self.data.append(simulations)
        
        # Load and aggregate all simulations from .pt files
        else:
            self.data = []
            for file in self.data_files:
                file_path = os.path.join(self.data_dir, file)
                simulations = torch.load(file_path, weights_only=False)
                self.data.extend(simulations)

    def __len__(self):
        # Return the total number of simulations
        return len(self.data)

    def __getitem__(self, idx):
        # Return the simulation at the given indeX
        return self.data[idx]
    
    def setup(self):
        """
        Sets up the edge_index and other necessary attributes for each graph data in the dataset.
        """
        if self.rollout:
            for sim in self.data:
                for data in sim:
                    # Variable extraction from raw data
                    node_index, total_loads = find_nonzero_indices(data.f)
                    w_edges = create_wedge_index(data.x[:, :3], node_index, self.att_radius)
                    u_m, u0_m, u_w, u0_w = set_u(data.x, data.q_0, data.edge_index, w_edges)
                    n0, _, n = label_all(node_index, data)

                    # Variable assignment to the data object
                    data.w_edge_index = w_edges
                    data.node_index = node_index
                    data.load_number = total_loads
                    data.total_f_u = torch.sum(torch.abs(data.f))
                    data.du_m = u_m - u0_m
                    data.du_w = u_w - u0_w
                    data.n0 = n0
                    data.n = n

        else:
            for data in self.data:
                # Variable extraction from raw data
                node_index, total_loads = find_nonzero_indices(data.f)
                w_edges = create_wedge_index(data.x[:, :3], node_index, self.att_radius)
                u_m, u0_m, u_w, u0_w = set_u(data.x, data.q_0, data.edge_index, w_edges)
                n0, _, n = label_all(node_index, data)

                # Variable assignment to the data object
                data.w_edge_index = w_edges
                data.node_index = node_index
                data.load_number = total_loads
                data.n0 = n0
                data.n = n
                data.total_f_u = torch.sum(torch.abs(data.f))
                # Compute the displacement
                data.u_m = u_m
                data.u_w = u_w
                data.u0_m = u0_m
                data.u0_w = u0_w
    
    def compute_statistics(self, variable_name, absolute=False):
        """Computes statistics (mean, std) for a given variable."""
        stack = [getattr(data, variable_name) for data in self.data]
        cat = torch.cat(stack, dim=0)
        cat = torch.abs(cat) if absolute else cat
        means, stds = torch.mean(cat, dim=0), torch.std(cat, dim=0)

        # Prevent division by zero
        stds = torch.where(stds == 0, torch.tensor(1.0, dtype=stds.dtype), stds)

        return {'means': means, 'stds': stds}