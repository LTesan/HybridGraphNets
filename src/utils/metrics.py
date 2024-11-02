import numpy as np
import torch

def rrmse_inf(data_ground_truth, data_predicted):
    """
    Calculate the Root Mean Squared Error (RMSE) normalized by the infinity norm.
    
    Please pay attention to the inputs, it must have the following shape:
        (snapshots, nodes, variables).
        E.g. (200, 550, 3) would be 200 snapshots, of a graph of 550 nodes and 3 variables, such as position.
        E.g. (200, 550, 1) would be 200 snapshots, of a graph of 550 nodes and 1 variable, such as S.Mises.
        Eg (550, 3) would be taken care of automatically and transformed as (1, 550, 3)
        E.g. (200, 550) WON'T WORK, because it will interpretate the input as 550 variable vector.

    Args:
        data_ground_truth (numpy.ndarray): Ground truth data with shape (snapshot, nodes, variable).
                                            In case of (nodes, variable), it expands snapshot dim to 1.
        data_predicted (numpy.ndarray): Predicted data with the same shape as data_ground_truth.
                                        In case of (nodes, variable), it expands snapshot dim to 1.

    Returns:
        float: RMSE normalized by the infinity norm.
    """

    x, y = data_ground_truth, data_predicted

    if isinstance(x, torch.Tensor):
        x = np.asarray(x)
        y = np.asarray(y)

    if len(x.shape) == 2:
        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)

    # Infinity norm calculation
    se_inf = []
    for i in range(x.shape[0]):
        x_snap = x[i]
        error = x[i] - y[i]
        infinite_norm_se = np.linalg.norm(x_snap, ord=np.inf)**2  + 1e-8
        for j in range(x_snap.shape[0]):
            l2_norm_se = np.linalg.norm(error[j], ord=2)**2
            se_inf.append(l2_norm_se / infinite_norm_se)
    # Cast into array
    mse_inf = np.mean(np.array(se_inf))
    # Square root of the mean squared error
    rmse_inf = np.sqrt(mse_inf)

    return rmse_inf

def rrmse_inf_timestep(data_ground_truth, data_predicted, time_frames_per_timestep=20):
    """
    Calculate the Root Mean Squared Error (RMSE) normalized by the infinity norm for each timestep.
    
    Please pay attention to the inputs, it must have the following shape:
        (total_time_frames, nodes, variables).
        E.g. (1260, 530, 3) would be 1260 time frames, of a graph of 530 nodes and 3 variables, such as position.
        E.g. (1260, 530, 1) would be 1260 time frames, of a graph of 530 nodes and 1 variable, such as S.Mises.
        Eg (530, 3) would be taken care of automatically and transformed as (1, 530, 3)
        E.g. (1260, 530) WON'T WORK, because it will interpret the input as 530 variable vector.

    Args:
        data_ground_truth (numpy.ndarray): Ground truth data with shape (total_time_frames, nodes, variables).
                                            In case of (nodes, variables), it expands total_time_frames dim to 1.
        data_predicted (numpy.ndarray): Predicted data with the same shape as data_ground_truth.
                                        In case of (nodes, variables), it expands total_time_frames dim to 1.
        time_frames_per_timestep (int): Number of time frames per timestep.

    Returns:
        list: RMSE normalized by the infinity norm for each timestep.
    """

    x, y = data_ground_truth, data_predicted

    if len(x.shape) == 2:
        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)

    # Reshape to group time frames into timesteps
    num_timesteps = x.shape[0] // time_frames_per_timestep
    x = x[:num_timesteps * time_frames_per_timestep].reshape(num_timesteps, time_frames_per_timestep, x.shape[1], x.shape[2])
    y = y[:num_timesteps * time_frames_per_timestep].reshape(num_timesteps, time_frames_per_timestep, y.shape[1], y.shape[2])

    # Calculating RRMSE per timestep
    rrmse_per_timestep = []

    for i in range(num_timesteps):
        rrmse_per_timestep.append(rrmse_inf(x[i], y[i]))
    
    return np.array(rrmse_per_timestep)

def rmse(x, y):
    """
    Calculates the Root Mean Squared Error (RMSE) between two arrays x and y.

    Parameters:
    x (array-like): Array of true values.
    y (array-like): Array of predicted values.

    Returns:
    float: The Root Mean Squared Error (RMSE) between x and y.
    """

    if isinstance(x, torch.Tensor):
        x = np.asarray(x)
        y = np.asarray(y)

    # Calculate squared error
    se = (x - y) ** 2
    # Calculate mean squared error
    mse = np.mean(se)

    return np.sqrt(mse)

def rmse(x, y):
    """
    Calculates the Root Mean Squared Error (RMSE) between two arrays x and y.

    Parameters:
    x (array-like): Array of true values.
    y (array-like): Array of predicted values.

    Returns:
    float: The Root Mean Squared Error (RMSE) between x and y.
    """
    # Calculate squared error
    se = (x - y) ** 2
    # Calculate mean squared error
    mse = se.mean()

    return np.sqrt(mse)

def mse_l2(data_ground_truth, data_predicted):
    """
    Calculate the Mean Relative Error normalized l2 norm.
    
    Please pay attention to the inputs, it must have the following shape:
        (snapshots, nodes, variables).
        E.g. (200, 550, 3) would be 200 snapshots, of a graph of 550 nodes and 3 variables, such as position.
        E.g. (200, 550, 1) would be 200 snapshots, of a graph of 550 nodes and 1 variable, such as S.Mises.
        Eg (550, 3) would be taken care of automatically and transformed as (1, 550, 3)
        E.g. (200, 550) WON'T WORK, because it will interpretate the input as 550 variable vector.

    Args:
        data_ground_truth (numpy.ndarray): Ground truth data with shape (snapshot, nodes, variable).
                                            In case of (nodes, variable), it expands snapshot dim to 1.
        data_predicted (numpy.ndarray): Predicted data with the same shape as data_ground_truth.
                                        In case of (nodes, variable), it expands snapshot dim to 1.

    Returns:
        float: RME normalized by the l2 norm.
    """

    x, y = data_ground_truth, data_predicted

    if isinstance(x, torch.Tensor):
        x = np.asarray(x)
        y = np.asarray(y)

    if len(x.shape) == 2:
        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)

    # L2 relative norm calculation
    e = x - y
    gt = x 

    # Position + Velocity + Stress Tensor
    L2_re = ((e[:,:,:]**2).sum((1,2)) / (gt[:,:,:]**2).sum((1,2)))**0.5

    # Mean L2 relative error
    L2_mre = np.mean(L2_re)

    return L2_mre, L2_re