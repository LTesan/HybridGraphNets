import argparse  # Module for parsing command-line arguments
import numpy as np  # Library for numerical operations
import torch  # PyTorch library for deep learning
import wandb  # Weights and Biases for experiment tracking
from src.solver.solver import Solver  # Importing the Solver class from the specified module

import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend for plotting


def parse_args():
    """
    Parse command-line arguments for training and testing the model.
    
    Returns:
        Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Training and testing control algorithm")

    # Adding arguments to the parser
    parser.add_argument("--name", type=str, default="Hybrid-GNN", help="Name of the run")
    parser.add_argument("--project", type=str, default="tester", help="WandB project name")
    parser.add_argument("--train", action="store_true", help="Flag to train the model")
    parser.add_argument("--n_hidden", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--dim_hidden", type=int, default=250, help="Dimension of hidden layers")
    parser.add_argument("--passes", type=int, default=12, help="Number of passes")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--att_radius", type=float, default=float('inf'), help="Attention radius")
    parser.add_argument("--noise_var", nargs='+', type=float, default=[0.001, 0.01], help="Noise constant values")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lambda_d", type=float, default=5., help="Lambda value for loss regularization")
    parser.add_argument("--max_epoch", type=int, default=6000, help="Maximum number of epochs")
    parser.add_argument("--pretrained", action="store_true", help="Flag to use pretrained model")
    parser.add_argument("--plot_rollout", action="store_true", help="Flag to plot rollout results")
    parser.add_argument("--node", type=int, default=0, help="Node index for plotting rollout results")

    return parser.parse_args()  # Return the parsed arguments


def main(args):
    """
    Main function to orchestrate the training and evaluation process.
    
    Args:
        args: Parsed command-line arguments.
    """
    torch.set_float32_matmul_precision('medium')  # Set precision for matrix multiplication in PyTorch

    # Initialize WandB
    try:
        wandb.login()  # Attempt to log in to WandB
    except Exception as e:
        print(f"Failed to log in to WandB: {e}")  # Print error message if login fails
        return

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)  # Set the seed for PyTorch
    np.random.seed(args.seed)  # Set the seed for NumPy

    # Initialize the solver
    solver = Solver(args)  # Create an instance of the Solver class with parsed arguments

    # Train the model if specified
    if args.train:
        try:
            solver.train()  # Call the train method of the solver
        except Exception as e:
            print(f"Training failed: {e}")  # Print error message if training fails
            return

    # Evaluate the model
    try:
        with torch.no_grad():  # Disable gradient calculation for evaluation
            solver.test()  # Call the test method of the solver
    except Exception as e:
        print(f"Testing failed: {e}")  # Print error message if testing fails


if __name__ == "__main__":
    args = parse_args()  # Parse command-line arguments
    main(args)  # Call the main function
