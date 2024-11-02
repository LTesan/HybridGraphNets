import torch


def label_all(node_index, data):
        """
        Labels the nodes and features for the given data.

        Parameters:
        data (Data): The data object containing node features (n) and edge features (f).

        Returns:
        tuple: A tuple containing the original node features (n0), a binary tensor indicating
            the presence of a feature (f0), and a concatenated tensor of node features and
            the binary feature tensor (n).
        """
        # Extract node features
        n0 = data.n
        
        # Create a binary tensor indicating the presence of features
        zeros_f = torch.zeros_like(data.f[:,0])
        zeros_f.scatter_(0,node_index,1)
        
        # Concatenate the original node features with the binary feature tensor
        n = torch.cat((n0, zeros_f.unsqueeze(1)), dim=1)
        
        return n0, zeros_f, n

def create_wedge_index(nodes, nodes_index, att_radius):
        """
        Create a new edge index based on the connectivity of a node with others within a specified radius.

        Parameters:
        nodes (list of tuple): List of 3D positions of nodes.
        node_index (int): Indexes of the nodes from which to measure the radius.
        att_radius (float): The radius within which to find connected nodes.

        Returns:
        list of tuple: List of tuples representing the edges.
        """

        # Reshape nodes to batched format
        load_size = 1

        loads = len(nodes_index) // load_size
        nodes_size = nodes.size(0) // load_size
        nodes_reshape = nodes.view(load_size, nodes_size, 3)
        selected_node_position = nodes[nodes_index, :].view(load_size, loads, 3)

        # Calculate the Euclidean distances from the selected nodes to all other nodes
        distances = torch.norm(nodes_reshape[:, None, :, :] - selected_node_position[:, :, None, :], dim=3)

        # Get boolean mask of nodes within the specified radius
        within_radius_mask = distances < att_radius

        # Get indices of nodes within the specified radius
        within_radius_indices = within_radius_mask.nonzero(as_tuple=False)

        # Create source indices (selected node index repeated)
        source_indices = nodes_index[within_radius_indices[:, 1] + within_radius_indices[:, 0] * loads]

        # Create the edge indices
        edge_index = torch.stack((
                source_indices,
                within_radius_indices[:, 2] + within_radius_indices[:, 0] * nodes_size
                ), dim=0)

        return edge_index

def find_nonzero_indices(tensor):
        """
        Find the indices of non-zero elements in a 3D tensor.
        
        Args:
        tensor (torch.Tensor): The 3-dimensional input tensor.
        
        Returns:
        torch.Tensor: Indices of non-zero elements in the tensor.
        """
        
        # Find non-zero indices in the entire tensor
        nonzero_indices = torch.nonzero(tensor,as_tuple=True)
        total_loads = len(torch.unique(nonzero_indices[0]))
        
        return torch.unique(nonzero_indices[0]), total_loads

def set_u(x, q0, edge_index, w_edge_index):
        x = x[:,:3]
        # Set the edge attributes for the mesh and world
        u_m = x[edge_index[0]] - x[edge_index[1]]
        u0_m = q0[edge_index[0]] - q0[edge_index[1]]
        u_w = x[w_edge_index[0]] - x[w_edge_index[1]]
        u0_w = q0[w_edge_index[0]] - q0[w_edge_index[1]]
        return u_m, u0_m, u_w, u0_w