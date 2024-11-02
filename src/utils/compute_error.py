import torch
import wandb
from src.utils.metrics import *
from src.utils.io import save_data
from src.utils.plots import plot_z_graphs, plot_z_graphs_noise

def evaluate_and_print(results, set_name='Test', save=True):
    print(f"[{set_name} Set Evaluation]")
    results = remove_initial_conditions(results)
    net, gt = collapse_results(results)
    print('Mean L2 relative error: ')
    error_L2, re_L2 = print_error_L2(net, gt)
    print('Mean inf relative error: ')
    error_inf, error_inf_timestep = print_error_inf(net, gt)
    print('Mean quadratic error: ')
    error = print_error(net, gt)
    wandb.log({
        f'L2_q_{set_name}': error_L2['q'],
        f'L2_v_{set_name}': error_L2['v'],
        f'L2_S_{set_name}': error_L2['sigma'],
        f'inf_q_{set_name}': error_inf['q'],
        f'inf_v_{set_name}': error_inf['v'],
        f'inf_S_{set_name}': error_inf['sigma'],
        f'quad_q_{set_name}': error['q'],
        f'quad_v_{set_name}': error['v'],
        f'quad_S_{set_name}': error['sigma']
    })
    print(f"[{set_name} Evaluation Finished]\n")
    if save:
        save_data(error_inf_timestep, f'outputs/test_statistics/{set_name}.json')
    mean = mean_error(error_inf)
    return error_inf_timestep, mean

def remove_initial_conditions(results):
    for key in results.keys():
        for idx, data in enumerate(results[key]):
            new = data[1:,:,:]
            results[key][idx] = new
    return results

def mean_error(error):
    mean = []
    for key in error.keys():
        mean.append(error[key])
    mean_value = sum(mean)/len(mean)
    return mean_value

def collapse_results(results):
    net_col = torch.cat(results['z_net'], dim=0)
    net = {'q': net_col[:, :, :3], 'v': net_col[:, :, 3:6], 'sigma': net_col[:, :, 6:]}
    gt_col = torch.cat(results['z_gt'], dim=0)
    gt = {'q': gt_col[:, :, :3], 'v': gt_col[:, :, 3:6], 'sigma': gt_col[:, :, 6:]}
    return net, gt

def print_error_L2(net, gt):
    """
    Print the mean L2 relative error for each state variable.

    Parameters:
    error (dict): Dictionary containing errors for different state variables.

    Returns:
    None
    """
    error_L2 = {'q': [], 'v': [], 'sigma': []}
    re_L2 = {'q': [], 'v': [], 'sigma': []}
    for key in error_L2.keys():
        error_L2[key], re_L2[key] = mse_l2(gt[key], net[key])
        print(f"Mean L2 relative error for {key}: {error_L2[key]:.6f}")
    return error_L2, re_L2
    

def print_error_inf(net, gt):
    """
    Print the mean inf relative error for each state variable.

    Parameters:
    error (dict): Dictionary containing errors for different state variables.

    Returns:
    None
    """
    
    error_inf = {'q': [], 'v': [], 'sigma': []}
    error_inf_timestep = {'q': [], 'v': [], 'sigma': []}
    for key in error_inf.keys():
        error_inf[key] = rrmse_inf(gt[key], net[key])
        error_inf_timestep[key] = rrmse_inf_timestep(gt[key], net[key])
        print(f"Inf root relative mean square error for {key}: {error_inf[key]:.6f}")
    return error_inf, error_inf_timestep

def print_error(net, gt):
    """
    Print the mean quad error for each state variable.

    Parameters:
    error (dict): Dictionary containing errors for different state variables.

    Returns:
    None
    """
    
    qerror = {'q': [], 'v': [], 'sigma': []}
    for key in qerror.keys():     
        qerror[key] = rmse(gt[key], net[key])
        print(f"Relative mean square {key} error: {qerror[key]:.6f}")
    return qerror

def plot_rollout_error(args, results, n, sim, node, noise, u_mean):
    # Plot the rollout 2D representation
    plot_z_graphs(results['z_net'][sim], results['z_gt'][sim], n[sim], node_val=node)
    plot_z_graphs_noise(results['z_net'][sim], results['z_gt'][sim], n[sim], noise_var=noise, u_mean=u_mean, node_val=node)