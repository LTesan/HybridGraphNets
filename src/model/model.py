import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.model.encoders import MLP, EdgeModelW, EdgeModelM, NodeModel, MetaLayer
from src.utils.pp import set_u
from torch_scatter import scatter_add

class TIGNN(pl.LightningModule):
    def __init__(self, args, stats, noise, lambda_d=20.):
        super(TIGNN, self).__init__()
        
        # Save arguments and statistics
        self.args = args
        self.stats = stats
        self.dim_z = 12  # Dimensionality of the output
        self.lambda_d = lambda_d  # Weight for degree loss
        self.noise = noise  # Noise settings
        self.param = stats['stats_u_m']['means']  # Parameters from stats

        # Save hyperparameters for easy checkpointing and logging
        self.save_hyperparameters(args)
        self.optimizer = torch.optim.Adam

        # Extract parameters from args
        self.passes = args.passes  # Number of processing passes
        self.lr = args.lr  # Learning rate
        n_hidden = args.n_hidden  # Number of hidden layers
        dim_hidden = args.dim_hidden  # Dimension of hidden layers
        self.batch_size = args.batch_size  # Batch size
        self.dt = 1/20  # Time step for integration

        # Initialize Encoder MLPs
        self.encoder_world = MLP([6] + n_hidden * [dim_hidden] + [dim_hidden])
        self.encoder_mesh = MLP([12] + n_hidden * [dim_hidden] + [dim_hidden])
        self.encoder_edgeW = MLP([8] + n_hidden * [dim_hidden] + [dim_hidden])
        self.encoder_edgeM = MLP([8] + n_hidden * [dim_hidden] + [dim_hidden])

        # Initialize Processor MLPs
        self.processor = nn.ModuleList()
        node_model = NodeModel(args)  # Initialize node model
        edge_modelw = EdgeModelW(args)  # Initialize edge model for world
        edge_modelm = EdgeModelM(args)  # Initialize edge model for mesh
        GraphNet = MetaLayer(node_model=node_model, edge_modelw=edge_modelw, edge_modelm=edge_modelm)
        
        # Append the processing layers
        for _ in range(self.passes):
            self.processor.append(GraphNet)

        # Initialize Decoder MLPs
        self.decoder_dE = MLP([dim_hidden] + n_hidden * [dim_hidden] + [self.dim_z])
        self.decoder_dS = MLP([dim_hidden] + n_hidden * [dim_hidden] + [self.dim_z])
        self.decoder_L = MLP([3 * dim_hidden] + n_hidden * [dim_hidden] + [int(self.dim_z * (self.dim_z + 1) / 2 - self.dim_z)])
        self.decoder_M = MLP([3 * dim_hidden] + n_hidden * [dim_hidden] + [int(self.dim_z * (self.dim_z + 1) / 2)])

        # Initialize matrices for later use
        diag = torch.eye(self.dim_z, self.dim_z)  # Identity matrix
        self.diag = diag[None]
        self.ones = torch.ones(self.dim_z, self.dim_z)  # Matrix of ones

        # Lists to save results during testing
        self.z_net_list = []
        self.z_gt_list = []
        self.n_list = []
        self.w_edge_list = []
        self.total_loads = []
        self.total_f_u = []
        self.deg_list = []

    def forward(self, x_w, x_m, edge_attr_w, edge_attr_m, edge_index_w, edge_index_m):

        # Encode
        x_m = self.encoder_mesh(x_m)
        x_w = self.encoder_world(x_w)

        src, dest = edge_index_m
        edge_attr_w = self.encoder_edgeW(edge_attr_w)
        edge_attr_m = self.encoder_edgeM(edge_attr_m)

        # Process
        for GraphNet in self.processor:
            x_res, edge_attr_res_w, edge_attr_res_m = GraphNet(x_m, x_w, edge_index_w, edge_index_m, edge_attr_w, edge_attr_m)
            x_m += x_res
            edge_attr_w += edge_attr_res_w
            edge_attr_m += edge_attr_res_m

        # Decode
        dzdt_net, loss_deg_E, loss_deg_S = self.nodal_decoder(x_m, edge_attr_m, src, dest)
        return dzdt_net, loss_deg_E, loss_deg_S

    def training_step(self, batch, batch_idx):
        # Define the training step (forward pass and loss calculation)
        z, z1, x_m, x_w, edge_attr_m, edge_attr_w = self.preprocess(batch)
        z_t_hat, loss_deg_E, loss_deg_S = self(x_w, x_m, edge_attr_w, edge_attr_m, batch.w_edge_index, batch.edge_index)
        z_t = (z1 - z)/self.dt
        loss = self.loss_function(z_t_hat, z_t)
        total_loss = loss + self.lambda_d * (loss_deg_E + loss_deg_S)
        self.log('train_loss', total_loss, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log('train_loss_deg_E', loss_deg_E, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log('train_loss_deg_S', loss_deg_S, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log('train_data_loss', loss, on_epoch=True, on_step=False, batch_size=self.batch_size)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Define the validation step (forward pass and loss calculation)
        z, z1, x_m, x_w, edge_attr_m, edge_attr_w = self.preprocess(batch)
        z_t_hat, loss_deg_E, loss_deg_S = self(x_w, x_m, edge_attr_w, edge_attr_m, batch.w_edge_index, batch.edge_index)
        z_t = (z1 - z)/self.dt
        loss = self.loss_function(z_t_hat, z_t)
        total_loss = loss + self.lambda_d * (loss_deg_E + loss_deg_S)
        self.log('val_loss', total_loss, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log('val_loss_deg_E', loss_deg_E, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log('val_loss_deg_S', loss_deg_S, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log('val_data_loss', loss, on_epoch=True, on_step=False, batch_size=self.batch_size)
        return total_loss
    
    def test_step(self, batch, batch_idx):
        # Define the test step (forward pass and loss calculation)
        z_net, z_gt, w_edges, n_value, sim_info, deg = self.integrate_sim(batch, full_rollout=True, test=True)
        self.z_net_list.append(z_net)
        self.z_gt_list.append(z_gt)
        self.w_edge_list.append(w_edges)
        self.n_list.append(n_value)
        self.total_loads.append(sim_info['total_loads'])
        self.total_f_u.append(sim_info['total_f_u'])
        self.deg_list.append(deg)
       

    def configure_optimizers(self):
        # Set up the optimizer
        optimizer = self.optimizer(self.parameters(), lr=self.lr)

        # Set up the learning rate scheduler - ReduceLROnPlateau
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=100, 
                threshold=0.0001, 
                min_lr=1e-8
            ),
            'monitor': 'train_loss',  # Monitor the validation loss
            'interval': 'epoch',
            'frequency': 1,
        }

        return [optimizer], [scheduler]

    def loss_function(self, y_hat, y):
        # Define a loss function, e.g., Mean Squared Error
        return nn.MSELoss()(y_hat, y)

    def normalize(self, x, stats):
        # Asegúrate de que 'means' y 'stds' estén en el mismo dispositivo que 'x'
        means = stats['means'].to(x.device)
        stds = stats['stds'].to(x.device)
        # Normaliza los datos de entrada
        return (x - means) / stds

    def denormalize(self, x, stats):
        # Asegúrate de que 'means' y 'stds' estén en el mismo dispositivo que 'x'
        means = stats['means'].to(x.device)
        stds = stats['stds'].to(x.device)
        # Desnormaliza los datos de entrada
        return x * stds + means
    
    def integrate_sim(self, data, full_rollout=True, test=False):
        """
        Integrates the simulation for a given dataset and data index.

        Parameters:
        dataset (list): The dataset containing the simulation data.
        data_index (int): The index of the specific data to simulate.
        WorldInfo_flag (bool): Whether to collect WorldInfo data. Default is False.
        full_rollout (bool): Whether to use full rollout mode. Default is True.

        Returns:
        tuple: z_net (tensor), z_gt (tensor), and optionally WorldInfo (list)
        """
        # Initialize WorldInfo list if the flag is set
        if test:
            sim_info = {'total_loads': data[0].load_number.detach().cpu().numpy(), 'total_f_u': data[0].total_f_u.detach().cpu().numpy()}
        WorldInfo = []
        n_value = data[0].n.detach().cpu()

        # Extract data list and initial conditions
        N_nodes = data[0].x.size(0)

        # Preallocate tensors for network output and ground truth
        z_net = torch.zeros(len(data) + 1, N_nodes, 12)
        z_gt = torch.zeros(len(data) + 1, N_nodes, 12)

        # Set initial conditions
        z_net[0] = data[0].x
        z_gt[0] = data[0].x

        # Itialize the state
        z = data[0].x.to(self.device)

        # Rollout loop through each time step
        for t, snap in enumerate(data):
            snap = snap.to(self.device)

            # Collect WorldInfo if the flag is set
            WorldInfo.append(snap.w_edge_index.detach().cpu())

            # Pre-process the data
            z, _, x_m, x_w, edge_attr_m, edge_attr_w = self.preprocess(snap, z=z, noise=False, integrate=True)
            # Perform a forward pass through the network and integrate
            z1_t_net, deg1, deg2 = self(x_w, x_m, edge_attr_w, edge_attr_m, snap.w_edge_index, snap.edge_index)
            z1_net = z +  z1_t_net*self.dt
            z1_net = self.denormalize(z1_net, self.stats['stats_z'])

            # Save the results
            z_net[t + 1] = z1_net
            z_gt[t + 1] = snap.y

            # Apply boundary conditions
            n = snap.n.detach().cpu()
            gt = snap.y.detach().cpu()

            bc1 = n[:, 1] == 1
            z_net[t + 1][bc1, :6] = gt[bc1][:, :6]

            bc2 = n[:, 2] == 1
            z_net[t + 1][bc2, :6] = gt[bc2][:, :6]

            # Update the state for the next step
            z = z1_net.detach() if full_rollout else snap.y
            deg = (deg1 + deg2).detach().cpu()

        # Return the results with or without WorldInfo
        if test:
            return z_net, z_gt, WorldInfo, n_value, sim_info, deg
        else:
            return z_net, z_gt, WorldInfo, n_value
    
    def add_noise(self, var, noise, n):
        """
        Adds relative noise to a tensor variable.

        Parameters:
        var (torch.Tensor): The tensor to which noise will be added.
        noise (float): The maximum relative noise as a fraction of the variable's value.

        Returns:
        torch.Tensor: The tensor with added noise.
        """
        # Bool
        bool_n = (n[:, 1] == 1) | (n[:, 2] == 1)
        bool_n = ~bool_n
        # Generate noise tensors
        noise_tensor_q = (torch.rand_like(var[:, :3]) * 2 - 1)*torch.tensor(noise[0]).to(self.device)
        noise_tensor_v = (torch.rand_like(var[:, 3:]) * 2 - 1)*torch.tensor(noise[1]).to(self.device)
        # Add noise to the variable
        noisy_var_q = var[:, :3]
        noisy_var_q[bool_n] += noise_tensor_q[bool_n] * self.param.to(self.device)
        noisy_var_v = var[:, 3:] + noise_tensor_v * var[:, 3:]
        noisy_var = torch.cat((noisy_var_q, noisy_var_v), dim=1)
        return noisy_var
    
    def preprocess(self, batch, z=None, noise=True, integrate=False):
        if integrate:
            z = z
        elif z is None:
            z = batch.x
        if noise:
            # Add noise to the input
            z = self.add_noise(batch.x, self.noise, batch.n)
        else:
            z = z
        # Set uw and um
        u_m, u0_m, u_w, u0_w = set_u(z, batch.q_0, batch.edge_index, batch.w_edge_index)
        
        # Norm the input
        z = self.normalize(z, self.stats['stats_z'])
        y = self.normalize(batch.y, self.stats['stats_z'])
        f = self.normalize(batch.f, self.stats['stats_f'])
        u_m = self.normalize(u_m, self.stats['stats_u_m'])
        u_w = self.normalize(u_w, self.stats['stats_u_w'])
        u0_m = self.normalize(u0_m, self.stats['stats_u_m'])
        u0_w = self.normalize(u0_w, self.stats['stats_u_w'])

        # Pre-process
        v = z[:, 3:]
        x_m = torch.cat((v, batch.n), dim=1)
        x_w = torch.cat((f, batch.n), dim=1)

        # Edge attributes mesh
        u_norm_m = torch.norm(u_m,dim=1).reshape(-1,1)
        u0_norm_m = torch.norm(u0_m,dim=1).reshape(-1,1)
        edge_attr_m = torch.cat((u_m,u_norm_m,u0_m,u0_norm_m), dim=1)

        # Edge attributes world
        u_norm_w = torch.norm(u_w,dim=1).reshape(-1,1)
        u0_norm_w = torch.norm(u0_w,dim=1).reshape(-1,1)
        edge_attr_w = torch.cat((u_w,u_norm_w,u0_w,u0_norm_w), dim=1)

        return z, y, x_m, x_w, edge_attr_m, edge_attr_w

    def nodal_decoder(self, x, edge_attr, src, dest):
        dEdz = self.decoder_dE(x).unsqueeze(-1)
        dSdz = self.decoder_dS(x).unsqueeze(-1)

        l = self.decoder_L(torch.cat([edge_attr, x[src], x[dest]], dim=1))
        m = self.decoder_M(torch.cat([edge_attr, x[src], x[dest]], dim=1))

        L = torch.zeros(edge_attr.size(0), self.dim_z, self.dim_z, device=l.device, dtype=l.dtype)
        M = torch.zeros(edge_attr.size(0), self.dim_z, self.dim_z, device=m.device, dtype=m.dtype)
        L[:, torch.tril(self.ones, - 1) == 1] = l
        M[:, torch.tril(self.ones) == 1] = m
        # import matplotlib.pyplot as plt; plt.imshow(M[0,:,:].cpu().detach().numpy(), cmap='gray')

        Ledges = torch.subtract(L, torch.transpose(L, 1, 2))
        Medges = torch.bmm(M, torch.transpose(M, 1, 2))

        edges_diag = src == dest
        edges_neigh = src != dest

        L_dEdz = torch.matmul(Ledges, dEdz[dest, :, :])
        M_dSdz = torch.matmul(Medges, dSdz[dest, :, :])

        tot = (torch.matmul(Ledges[edges_diag, ...], dEdz) + torch.matmul(Medges[edges_diag, ...], dSdz))
        L_dEdz_M_dSdz = L_dEdz + M_dSdz

        dzdt_net = tot[..., 0] - scatter_add(L_dEdz_M_dSdz[..., 0][edges_neigh, ...], src[edges_neigh], dim=0)
        loss_deg_E = (torch.matmul(Medges[edges_diag, ...], dEdz)[..., 0] ** 2).mean()
        loss_deg_S = (torch.matmul(Ledges[edges_diag, ...], dSdz)[..., 0] ** 2).mean()

        return dzdt_net, loss_deg_E, loss_deg_S
if __name__ == '__main__':
    pass