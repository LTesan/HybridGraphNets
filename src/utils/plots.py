import matplotlib.pyplot as plt
import torch 
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib import animation
import wandb
import os
from matplotlib.patches import Patch

def plot_3D(z_net, z_gt, n, save_dir = 'outputs/test_statistics/', name='Liver_actuator', idx=0):
    """
    This function creates a 3D plot comparing Data Driven MeshGraphs predictions with ground truth.

    Parameters:
    z_net (torch.Tensor): Predictions from the GNN.
    z_gt (torch.Tensor): Ground truth data.
    data_list (list): List containing data.
    output_dir (str): Directory to save the plot.

    Returns:
    None
    """
    T = z_net.size(0)

    # Plot initialization
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.set_title('Data Driven GNN', fontsize=18, fontfamily='serif'), ax1.grid()
    ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
    ax1.view_init(elev=20, azim=40)
    ax2.set_title('Ground Truth', fontsize=18, fontfamily='serif'), ax2.grid()
    ax2.set_xlabel('X'), ax2.set_ylabel('Y'), ax2.set_zlabel('Z')
    ax2.view_init(elev=20, azim=40)
    # Delete ticks
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])
    # Adjust ranges
    var_gt, var_net =  z_gt[:,:,6], z_net[:,:,6]
    X, Y, Z = z_gt[:,:,0].numpy(), z_gt[:,:,1].numpy(), z_gt[:,:,2].numpy()
    tensor_max = torch.max(var_gt, var_net)
    tensor_min = torch.min(var_gt, var_net)
    z_min, z_max = tensor_min.min(), tensor_max.max()
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

    # Initial snapshot
    q1_net, q2_net, q3_net = z_net[0,:,0], z_net[0,:,1], z_net[0,:,2]
    q1_gt, q2_gt, q3_gt = z_gt[0,:,0], z_gt[0,:,1], z_gt[0,:,2]
    var_net0, var_gt0 = var_net[0,:], var_gt[0,:]
    # Bounding box
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax1.plot([xb], [yb], [zb], 'w')
        ax2.plot([xb], [yb], [zb], 'w')

    # Scatter points
    s1 = ax1.scatter(q1_net[n[:,0]==1], q2_net[n[:,0]==1], q3_net[n[:,0]==1], c=var_net0[n[:,0]==1], cmap='plasma', vmax=z_max, vmin=z_min)
    ax1.scatter(q1_net[n[:,1]==1], q2_net[n[:,1]==1], q3_net[n[:,1]==1], color='k')
    s2 = ax2.scatter(q1_gt[n[:,0]==1], q2_gt[n[:,0]==1], q3_gt[n[:,0]==1], c=var_gt0[n[:,0]==1], cmap='plasma', vmax=z_max, vmin=z_min)
    ax2.scatter(q1_gt[n[:,1]==1], q2_gt[n[:,1]==1], q3_gt[n[:,1]==1], color='k')    
    
    def scientific_notation(x, pos):
        """
        Convert the tick labels of the colorbar to scientific notation.
        """
        if x == 0:
            return "0"
        else:
            return "{:.0f}e4".format(x / 1e4)

    # Colorbar
    cbar1 = fig.colorbar(s1, ax=ax1, location='bottom', pad=0.08)
    cbar2 = fig.colorbar(s2, ax=ax2, location='bottom', pad=0.08)

    cbar1.set_label(r'$S_{11}$', fontsize=14, labelpad=10)
    cbar2.set_label(r'$S_{11}$', fontsize=14, labelpad=10)

    # Apply scientific notation formatter to the colorbars
    cbar1.ax.xaxis.set_major_formatter(FuncFormatter(scientific_notation))
    cbar2.ax.xaxis.set_major_formatter(FuncFormatter(scientific_notation))

    # Adjust tick label size and padding
    cbar1.ax.tick_params(labelsize=14, pad=12)
    cbar2.ax.tick_params(labelsize=14, pad=12)
    
    # Animation
    def animate(snap):
        ax1.cla()
        ax2.cla()
        # Delete ticks
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_zticklabels([])
        # Set title and labels
        ax1.set_title('Data Driven GNN', fontsize=18, fontfamily='serif'), ax1.grid()
        ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
        ax2.set_title('Ground Truth', fontsize=18, fontfamily='serif'), ax2.grid()
        ax2.set_xlabel('X'), ax2.set_ylabel('Y'), ax2.set_zlabel('Z')

        # Bounding box
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax1.plot([xb], [yb], [zb], 'w')
            ax2.plot([xb], [yb], [zb], 'w')

        # Scatter points
        q1_net, q2_net, q3_net = z_net[snap,:,0], z_net[snap,:,1], z_net[snap,:,2]
        q1_gt, q2_gt, q3_gt = z_gt[snap,:,0], z_gt[snap,:,1], z_gt[snap,:,2]
        var_nett, var_gtt =  var_net[snap,:], var_gt[snap,:]
        ax1.scatter(q1_net[n[:,0]==1], q2_net[n[:,0]==1], q3_net[n[:,0]==1], c=var_nett[n[:,0]==1], cmap='plasma', vmax=z_max, vmin=z_min)
        ax1.scatter(q1_net[n[:,1]==1], q2_net[n[:,1]==1], q3_net[n[:,1]==1], color='k')
        ax2.scatter(q1_gt[n[:,0]==1], q2_gt[n[:,0]==1], q3_gt[n[:,0]==1], c=var_gtt[n[:,0]==1], cmap='plasma', vmax=z_max, vmin=z_min)
        ax2.scatter(q1_gt[n[:,1]==1], q2_gt[n[:,1]==1], q3_gt[n[:,1]==1], color='k')

        return fig,
    anim = animation.FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=20)

    # Save as gif
    save_dir = save_dir + name + '_3Dplot.gif'
    anim.save(save_dir, writer=writergif)
    wandb.log({f'{name}_3Dplot': wandb.Image(save_dir)})
    plt.close()

def plot_connectivity_3D(z_net, z_gt, n, edge_index_list, save_dir='outputs/test_statistics/', name='Liver_actuator', idx=0, gt_flag=True):
    T = z_net.size(0)-1
    
    # Unpack edge indices
    src, dest = edge_index_list[0].cpu().numpy()

    # Plot initialization
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.set_title('Data Driven GNN', fontsize=18, fontfamily='serif')
    ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
    ax1.view_init(elev=20, azim=40)
    # Delete ticks
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    # Adjust ranges
    X, Y, Z = z_gt[:, :, 0].numpy(), z_gt[:, :, 1].numpy(), z_gt[:, :, 2].numpy()
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

    # Initial snapshot
    if gt_flag: q1_net, q2_net, q3_net = z_gt[0, :, 0], z_gt[0, :, 1], z_gt[0, :, 2]
    else: q1_net, q2_net, q3_net = z_net[0, :, 0], z_net[0, :, 1], z_net[0, :, 2]

    # Bounding box
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax1.plot([xb], [yb], [zb], 'w')

    # Scatter points
    s1 = ax1.scatter(q1_net[n[:, 0] == 1], q2_net[n[:, 0] == 1], q3_net[n[:, 0] == 1])
    ax1.scatter(q1_net[n[:, 1] == 1], q2_net[n[:, 1] == 1], q3_net[n[:, 1] == 1], color='k')

    # Generar colores únicos
    colors = ['g', 'r', 'b']

    # Asegúrate de que hay suficientes colores para los valores únicos en src
    unique_src = np.unique(src)
    assert len(unique_src) <= len(colors), "Hay más valores únicos en 'src' que colores disponibles."

    # Crear un diccionario para mapear cada valor de src a un color
    src_colors = {value: colors[idx] for idx, value in enumerate(unique_src)}

    # Dibujar las conexiones
    for i in range(len(src)):
        ax1.plot([q1_net[src[i]], q1_net[dest[i]]],
                [q2_net[src[i]], q2_net[dest[i]]],
                [q3_net[src[i]], q3_net[dest[i]]],
                color=src_colors[src[i]], alpha=0.8)

    # Animation
    def animate(snap):
        ax1.cla()

        # Delete ticks
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])

        # Unpack edge indices
        src, dest = edge_index_list[snap].cpu().numpy()
        
        # Set title and labels
        ax1.set_title('Data Driven GNN', fontsize=18, fontfamily='serif')
        ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')

        # Bounding box
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax1.plot([xb], [yb], [zb], 'w')

        # Scatter points
        if gt_flag: q1_net, q2_net, q3_net = z_gt[snap, :, 0], z_gt[snap, :, 1], z_gt[snap, :, 2]
        else: q1_net, q2_net, q3_net = z_net[snap, :, 0], z_net[snap, :, 1], z_net[snap, :, 2]

        ax1.scatter(q1_net[n[:, 0] == 1], q2_net[n[:, 0] == 1], q3_net[n[:, 0] == 1], color='lightblue')
        ax1.scatter(q1_net[n[:, 1] == 1], q2_net[n[:, 1] == 1], q3_net[n[:, 1] == 1], color='k')

        # Draw connectivity
        for i in range(len(src)):
            ax1.plot([q1_net[src[i]], q1_net[dest[i]]],
                     [q2_net[src[i]], q2_net[dest[i]]],
                     [q3_net[src[i]], q3_net[dest[i]]],
                     color=src_colors[src[i]], alpha=0.8)

        return fig,

    anim = animation.FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=20)

    save_dir = save_dir + name + f'_ConnPlot.gif'
    # Save as gif
    anim.save(save_dir, writer=writergif)
    wandb.log({f'{name}_ConnPlot': wandb.Image(save_dir)})
    plt.close()


def create_plots(z_net_list, z_gt_list, n_list, edge_index_list, name,  max_plots=1):
    """
    This function creates a specified number of random plots from the provided lists of network predictions,
    ground truth data, node information, and edge indices.

    Parameters:
    z_net_list (list of torch.Tensor): List of predictions from the GNN.
    z_gt_list (list of torch.Tensor): List of ground truth data.
    n_list (list of torch.Tensor): List of node information.
    edge_index_list (list of torch.Tensor): List of edge indices.
    max_plots (int): Maximum number of plots to create. Default is 1.

    Returns:
    None
    """

    # Create a list of random indices for the number of plots to generate
    random_indices = np.random.choice(len(z_net_list), max_plots, replace=False)

    # Loop through the random indices and generate plots
    for idx in random_indices:
        # Generate a 3D plot comparing GNN predictions with ground truth
        plot_3D(z_net_list[idx], z_gt_list[idx], n_list[idx], name=name, idx=idx)
        
        # Generate a 3D plot showing the connectivity of the GNN predictions
        plot_connectivity_3D(z_net_list[idx], z_gt_list[idx], n_list[idx], edge_index_list[idx], name=name, idx=idx)
    
    print(f"Plots saved in directory.")

def boxplot_error(test_error, save_path='outputs/test_statistics/', name='Liver_actuator'):
    """
    Create a boxplot comparing error distributions for different state variables in the test set,
    with individual data points plotted as dots to the left of each boxplot.

    Parameters:
    test_error (dict): Dictionary containing testing errors for different state variables.
    save_path (str): Path to save the boxplot image. Default is 'outputs/test_statistics/boxplot.png'.

    Returns:
    None
    """
    
    # Create a new figure with adjusted size for a longer and thinner plot
    fig, ax = plt.subplots(figsize=(3, 4))  # Longer and thinner figure

    # Create a list of error values for each state variable for testing
    state_variables_plot = [r'$q$', r'$\dot{q}$', r'$\sigma$']
    state_variables = list(test_error.keys())
    test_error_values = [test_error[key] for key in state_variables]

    # Define positions for the boxplots and scatter points
    positions = [1, 1.5, 2]
    scatter_offset = -0.15  # Offset for the scatter points to the left of the boxplots

    # Define colors for each state variable
    colors = ['black', 'darkblue', 'darkred']
    
    # Create boxplots for testing errors with logarithmic scale on the y-axis
    for pos, values, color in zip(positions, test_error_values, colors):
        bp = ax.boxplot(values, positions=[pos], widths=0.15, patch_artist=True, 
                        boxprops=dict(facecolor='white', edgecolor=color, linewidth=2), showfliers=False)
        ax.scatter(np.full_like(values, pos + scatter_offset), values, color=color, alpha=0.6, edgecolor='w', zorder=3)

    ax.set_yscale('log')  # Set y-axis to log scale (base 10)
    ax.tick_params(axis='y', labelsize=12)  # Smaller y-axis labels
    
    # Customize font size and family
    ax.set_xlabel('State Variables', fontsize=16, fontfamily='serif')
    
    # Adjust x-axis ticks and labels for better spacing
    ax.set_xticks(positions)
    ax.set_xticklabels(state_variables_plot, fontsize=16, fontfamily='serif', fontweight='bold', fontstyle='italic')
    
    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=1.0)

    # Customize the plot background
    ax.set_facecolor('whitesmoke')

    # Save the boxplot image with a transparent background
    save_path = save_path + name + '_boxplot.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', transparent=True)
    wandb.log({f'boxplot_unitary': wandb.Image(save_path)})
    plt.close()

def plot_z_graphs(z_net, z_gt, n, node_val=0, save_path = 'outputs/model_check/features_check.png', graph_path = 'outputs/model_check/graph_check.png'):
    """
    Plot various features (position, velocity, stress) comparison between ground truth and network predictions,
    and visualize the node's position in a 2D plot.

    Parameters:
    z_net (numpy.ndarray): Network predictions.
    z_gt (numpy.ndarray): Ground truth data.
    data_list (list): List of data.
    node_val (int): Index of the node to plot. Default is 0.
    output (str): Directory path to save the plots. Default is 'outputs/model_check'.

    Returns:
    None
    """

    z_net = z_net.numpy()
    z_gt = z_gt.numpy()
    # Define shades of black and red as RGB tuples
    black_shades = [(1, 0, 0), (0.5, 0, 0), (0, 0, 0)]  # Three tones of black
    red_shades = [(1, 0, 0), (0.5, 0, 0), (0, 0, 0)]  # Three tones of red
    timeframes = z_net.shape[0]
    timeframes = np.arange(1,timeframes+1)

    if node_val<n.shape[0]:
        plt.figure(figsize=(24, 8))

        # 2D representation of q
        plt.subplot(1, 4, 1)
        # Net
        plt.plot(timeframes, z_net[:, node_val, 0], label='q_x net', color=black_shades[0])
        plt.plot(timeframes, z_net[:, node_val, 1], label='q_y net', color=black_shades[1])
        plt.plot(timeframes, z_net[:, node_val, 2], label='q_z net', color=black_shades[2])
        # GT
        plt.plot(timeframes, z_gt[:, node_val, 0], label='q_x gt', linestyle='--', color=red_shades[0])
        plt.plot(timeframes, z_gt[:, node_val, 1], label='q_y gt', linestyle='--', color=red_shades[1])
        plt.plot(timeframes, z_gt[:, node_val, 2], label='q_z gt', linestyle='--', color=red_shades[2])
        # Add labels and title
        plt.xlabel('Timeframes', fontsize=16, fontfamily='serif')
        plt.ylabel('Position (m)', fontsize=16, fontfamily='serif')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=16)

        # 2D representation of v
        plt.subplot(1, 4, 2)
        # Net
        plt.plot(timeframes, z_net[:, node_val, 3], label='v_x net', color=black_shades[0])
        plt.plot(timeframes, z_net[:, node_val, 4], label='v_y net', color=black_shades[1])
        plt.plot(timeframes, z_net[:, node_val, 5], label='v_z net', color=black_shades[2])
        # GT
        plt.plot(timeframes, z_gt[:, node_val, 3], label='v_x gt', linestyle='--', color=red_shades[0])
        plt.plot(timeframes, z_gt[:, node_val, 4], label='v_y gt', linestyle='--', color=red_shades[1])
        plt.plot(timeframes, z_gt[:, node_val, 5], label='v_z gt', linestyle='--', color=red_shades[2])
        # Add labels and title
        plt.xlabel('Timeframes', fontsize=16, fontfamily='serif')
        plt.ylabel('Velocity (m/s)', fontsize=16, fontfamily='serif')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=16)

        # 2D representation of v
        plt.subplot(1, 4, 3)
        # Net
        plt.plot(timeframes, z_net[:, node_val, 6], label='S11_x net', color=black_shades[0])
        plt.plot(timeframes, z_net[:, node_val, 7], label='S22_y net', color=black_shades[1])
        plt.plot(timeframes, z_net[:, node_val, 8], label='S33_z net', color=black_shades[2])
        # GT
        plt.plot(timeframes, z_gt[:, node_val, 6], label='S11_x gt', linestyle='--', color=red_shades[0])
        plt.plot(timeframes, z_gt[:, node_val, 7], label='S22_y gt', linestyle='--', color=red_shades[1])
        plt.plot(timeframes, z_gt[:, node_val, 8], label='S33_z gt', linestyle='--', color=red_shades[2])
        # Add labels and title
        plt.xlabel('Timeframes', fontsize=16, fontfamily='serif')
        plt.ylabel('Stress (Pa)', fontsize=16, fontfamily='serif')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=16)

        # 2D representation of v
        plt.subplot(1, 4, 4)
        # Net
        plt.plot(timeframes, z_net[:, node_val, 9], label='S12_x net', color=black_shades[0])
        plt.plot(timeframes, z_net[:, node_val, 10], label='S23_y net', color=black_shades[1])
        plt.plot(timeframes, z_net[:, node_val, 11], label='S31_z net', color=black_shades[2])
        # GT
        plt.plot(timeframes, z_gt[:, node_val, 9], label='S12_x gt', linestyle='--', color=red_shades[0])
        plt.plot(timeframes, z_gt[:, node_val, 10], label='S23_y gt', linestyle='--', color=red_shades[1])
        plt.plot(timeframes, z_gt[:, node_val, 11], label='S31_z gt', linestyle='--', color=red_shades[2])
        # Add labels and title
        plt.xlabel('Timeframes', fontsize=16, fontfamily='serif')
        plt.ylabel('Stress (Pa)', fontsize=16, fontfamily='serif')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=16)

        plt.tight_layout()

        # Save plot
        plt.savefig(save_path)
        print(f'Gráfica features_check guardada en: {save_path}')

        # Plot 3D figure for awareness of the node 
        fig = plt.figure(figsize=(5, 8))
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        ax1.set_title('Features_Check_node_awareness'), ax1.grid()
        ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
        ax1.view_init(elev=20, azim=40)
        X, Y, Z = z_gt[:,:,0], z_gt[:,:,1], z_gt[:,:,2]
        q1_gt, q2_gt, q3_gt = z_gt[0,:,0], z_gt[0,:,1], z_gt[0,:,2]
        # Scatter points
        ax1.scatter(q1_gt[n[:,0]==1], q2_gt[n[:,0]==1], q3_gt[n[:,0]==1], color=(0.6, 0.6, 0.6))
        ax1.scatter(q1_gt[n[:,1]==1], q2_gt[n[:,1]==1], q3_gt[n[:,1]==1], color='k')
        ax1.scatter(q1_gt[node_val], q2_gt[node_val], q3_gt[node_val], label='Node plotted in features_check', marker='*', color='r', s=300)
        # Save plot
        plt.savefig(graph_path)
        wandb.log({f'features_check': wandb.Image(save_path)})
        plt.close()
    else: 
        print(f"Index {node_val} does not exist in the list.")

def plot_z_graphs_noise(z_net, z_gt, n, noise_var, u_mean, node_val=0, save_path = 'outputs/model_check/features_check_noise.png'):
    """
    Plot various features (position, velocity, stress) comparison between ground truth and network predictions in normalized space,
    and visualize the node's position in a 3D plot with confidence intervals considering noise.

    Parameters:
    z_net (numpy.ndarray): Network predictions.
    z_gt (numpy.ndarray): Ground truth data.
    data_list (list): List of data.
    noise_var (numpy.ndarray): Constant noise values.
    node_val (int): Index of the node to plot. Default is 0.
    output (str): Directory path to save the plots. Default is 'outputs/model_check'.

    Returns:
    None
    """

    z_net = z_net.numpy()
    z_gt = z_gt.numpy()
    # Define shades of black and red as RGB tuples
    black_shades = [(1, 0, 0), (0.5, 0, 0), (0, 0, 0)]  # Three tones of black
    red_shades = [(1, 0, 0), (0.5, 0, 0), (0, 0, 0)]  # Three tones of red
    timeframes = z_net.shape[0]
    timeframes = np.arange(1,timeframes+1)
    # Noise values
    u_mean = u_mean.numpy()
    errorValues = np.empty((12))
    errorValues[:3] = noise_var[0]
    errorValues[3:6] = noise_var[1]
    errorValues[6:] = noise_var[1]

    if node_val<n.shape[0]:
        plt.figure(figsize=(24, 8))

        # 2D representation of q
        plt.subplot(1, 4, 1)
        # Net
        plt.plot(timeframes, z_net[:, node_val, 0], label='q_x net', color=black_shades[0])
        plt.plot(timeframes, z_net[:, node_val, 1], label='q_y net', color=black_shades[1])
        plt.plot(timeframes, z_net[:, node_val, 2], label='q_z net', color=black_shades[2])
        # GT
        plt.plot(timeframes, z_gt[:, node_val, 0], label='q_x gt', linestyle='--', color=red_shades[0])
        plt.plot(timeframes, z_gt[:, node_val, 1], label='q_y gt', linestyle='--', color=red_shades[1])
        plt.plot(timeframes, z_gt[:, node_val, 2], label='q_z gt', linestyle='--', color=red_shades[2])
        # Add noise area
        for i in range(3):
            upper_bound = z_gt[:, node_val, i] + errorValues[i]*u_mean[i]
            lower_bound = z_gt[:, node_val, i] - errorValues[i]*u_mean[i]
            plt.fill_between(timeframes, upper_bound, lower_bound, color=black_shades[i], alpha=0.15)
        
        # Add labels and title
        plt.xlabel('Timeframes', fontsize=16, fontfamily='serif')
        plt.ylabel('Position (m)', fontsize=16, fontfamily='serif')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=16)

        # 2D representation of v
        plt.subplot(1, 4, 2)
        # Net
        plt.plot(timeframes, z_net[:, node_val, 3], label='v_x net', color=black_shades[0])
        plt.plot(timeframes, z_net[:, node_val, 4], label='v_y net', color=black_shades[1])
        plt.plot(timeframes, z_net[:, node_val, 5], label='v_z net', color=black_shades[2])
        # GT
        plt.plot(timeframes, z_gt[:, node_val, 3], label='v_x gt', linestyle='--', color=red_shades[0])
        plt.plot(timeframes, z_gt[:, node_val, 4], label='v_y gt', linestyle='--', color=red_shades[1])
        plt.plot(timeframes, z_gt[:, node_val, 5], label='v_z gt', linestyle='--', color=red_shades[2])
        # Add noise area
        for i in range(3,6):
            upper_bound = z_gt[:, node_val, i] + errorValues[i]*z_gt[:, node_val, i]
            lower_bound = z_gt[:, node_val, i] - errorValues[i]*z_gt[:, node_val, i]
            plt.fill_between(timeframes, upper_bound, lower_bound, color=black_shades[i-3], alpha=0.15)
        # Add labels and title
        plt.xlabel('Timeframes', fontsize=16, fontfamily='serif')
        plt.ylabel('Velocity (m/s)', fontsize=16, fontfamily='serif')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=16)

        # 2D representation of S
        plt.subplot(1, 4, 3)
        # Net
        plt.plot(timeframes, z_net[:, node_val, 6], label='S11_x net', color=black_shades[0])
        plt.plot(timeframes, z_net[:, node_val, 7], label='S22_y net', color=black_shades[1])
        plt.plot(timeframes, z_net[:, node_val, 8], label='S33_z net', color=black_shades[2])
        # GT
        plt.plot(timeframes, z_gt[:, node_val, 6], label='S11_x gt', linestyle='--', color=red_shades[0])
        plt.plot(timeframes, z_gt[:, node_val, 7], label='S22_y gt', linestyle='--', color=red_shades[1])
        plt.plot(timeframes, z_gt[:, node_val, 8], label='S33_z gt', linestyle='--', color=red_shades[2])
        # Add noise area
        for i in range(6,9):
            upper_bound = z_gt[:, node_val, i] + errorValues[i]*z_gt[:, node_val, i]
            lower_bound = z_gt[:, node_val, i] - errorValues[i]*z_gt[:, node_val, i]
            plt.fill_between(timeframes, upper_bound, lower_bound, color=black_shades[i-6], alpha=0.15)
        # Add labels and title
        plt.xlabel('Timeframes', fontsize=16, fontfamily='serif')
        plt.ylabel('Stress (Pa)', fontsize=16, fontfamily='serif')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=16)

        # 2D representation of S tangential
        plt.subplot(1, 4, 4)
        # Net
        plt.plot(timeframes, z_net[:, node_val, 9], label='S12_x net', color=black_shades[0])
        plt.plot(timeframes, z_net[:, node_val, 10], label='S23_y net', color=black_shades[1])
        plt.plot(timeframes, z_net[:, node_val, 11], label='S31_z net', color=black_shades[2])
        # GT
        plt.plot(timeframes, z_gt[:, node_val, 9], label='S12_x gt', linestyle='--', color=red_shades[0])
        plt.plot(timeframes, z_gt[:, node_val, 10], label='S23_y gt', linestyle='--', color=red_shades[1])
        plt.plot(timeframes, z_gt[:, node_val, 11], label='S31_z gt', linestyle='--', color=red_shades[2])
        # Add noise area
        for i in range(9,12):
            upper_bound = z_gt[:, node_val, i] + errorValues[i]*z_gt[:, node_val, i]
            lower_bound = z_gt[:, node_val, i] - errorValues[i]*z_gt[:, node_val, i]
            plt.fill_between(timeframes, upper_bound, lower_bound, color=black_shades[i-9], alpha=0.15)
        # Add labels and title
        plt.xlabel('Timeframes', fontsize=16, fontfamily='serif')
        plt.ylabel('Stress (Pa)', fontsize=16, fontfamily='serif')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=16)

        plt.tight_layout()

        # Save plot
        plt.savefig(save_path)
        wandb.log({f'features_check_noise': wandb.Image(save_path)})
        plt.close()
        print(f'Gráfica features_check guardada en: {save_path}')
    else:
        print(f"Index {node_val} does not exist in the list.")


def plot_2D_ca(error, total_loads, total_f_u, save_path='outputs/test_statistics/'):
    """
    Function to create subplots for 2D scatter plots with errors of type 'q', 'v', and 'sigma'.
    
    Arguments:
    error -- A dictionary containing the keys 'q', 'v', and 'sigma', each of which
             corresponds to the error values for the scatter plot.
    total_loads -- List of total loads for each point.
    total_f_u -- List of total displacements (total U) for each point.
    save_path -- Path to save the output plots (default: 'outputs/test_statistics/').
    """
    # Stack the data together
    total_u = np.hstack(total_f_u)
    total_loads = np.hstack(total_loads)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns of subplots

    # Define the keys to plot
    error_keys = ['q', 'v', 'sigma']
    titles = ['Error for q', 'Error for v', 'Error for sigma']

    for i, key in enumerate(error_keys):
        # Create scatter plot for each error type
        scatter = axes[i].scatter(total_loads, total_u, c=error[key], cmap='coolwarm')
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=axes[i])
        cbar.set_label(f'{key} Error', fontsize=12)
        
        # Set axis labels and title for each subplot
        axes[i].set_xlabel('Total Loads', fontsize=12)
        axes[i].set_ylabel('Total U', fontsize=12)
        axes[i].set_title(titles[i], fontsize=14)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Ensure save_path exists and save the plot
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, '2D_ca_test.png'))  # Safe path handling
    
    # Close the plot to free up memory
    plt.close()