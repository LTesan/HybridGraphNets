import pytorch_lightning as pl
import torch
import os
import shutil
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor
from src.dataloader.dataModule import GraphDataModule
from src.model.model import TIGNN
from src.utils.compute_error import evaluate_and_print, plot_rollout_error
from src.model.callbacks import ValidationRolloutCallback
from src.utils.plots import create_plots, boxplot_error, plot_2D_ca

class Solver:
    def __init__(self, args):
        """
        Initializes the Solver with the provided arguments.
        
        Args:
            args (object): Command line arguments or configuration settings.
        """
        self.args = args
        self.name = args.name
        self._init_training_params(args)  # Initialize training parameters
        wandb.init(project=args.project, name=args.name)  # Initialize Wandb for logging
        self.logger = WandbLogger(args.project, args.name)  # Set up Wandb logger

        self._load_datasets()  # Load datasets for training, validation, and testing
        if self.pretrained:
            self._init_pretrained_net()  # Initialize pretrained network if specified
        else:
            self._init_net()  # Initialize new network
        self._load_trainer()  # Load the PyTorch Lightning Trainer
    
    def _load_datasets(self):
        """
        Loads the datasets for training, validation, and testing.
        """
        # Initialize the DataModule
        print("Loading datasets...")
        data_module = GraphDataModule(self.args, batch_size=self.batch_size)
        self.stats = data_module.setup()  # Set up the data module and get statistics

        # Load the DataLoaders
        self.train_loader = data_module.train_dataloader()  # Training DataLoader
        self.valid_loader = data_module.val_dataloader()  # Validation DataLoader
        self.test_loader = data_module.test_dataloader()  # Testing DataLoader
        print("Datasets loaded.")
    
    def _init_training_params(self, args):
        """
        Initializes the training parameters.
        
        Args:
            args (object): Command line arguments or configuration settings.
        """
        self.epochs = args.max_epoch  # Maximum number of epochs for training
        self.batch_size = args.batch_size  # Batch size for training
        self.lr = args.lr  # Learning rate for the optimizer
        self.lambda_d = args.lambda_d  # Regularization parameter
        self.att_radius = args.att_radius  # Attention radius for the model
        self.noise_var = args.noise_var  # Noise variance for data augmentation
        self.pretrained = args.pretrained  # Flag to indicate if a pretrained model should be used
        self.plot_rollout = args.plot_rollout  # Flag to indicate if rollout error should be plotted
        self.node = args.node  # Node index for specific operations

    def _load_trainer(self):
        """
        Initializes the PyTorch Lightning Trainer with the specified callbacks and settings.
        """
        # Define the save directory path using self.name
        save_dir = os.path.join('outputs/saved_models', self.name)

        # Check if the folder exists, and if it does, clear its contents
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)  # Remove the entire folder and its contents
        os.makedirs(save_dir, exist_ok=True)  # Create a clean directory

        # Define the checkpoint callback to save the model weights every 100 epochs
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir,  # Directory to save the model weights, includes self.name
            filename='model_epoch_{epoch}',  # Filename format, will include epoch number
            every_n_epochs=100,  # Save every 100 epochs
            save_top_k=-1,  # Save all checkpoints, not just the best one
        )

        # Define the custom validation rollout callback
        val_rollout_callback = ValidationRolloutCallback(self.test_loader, interval=100)
        
        # Create the LearningRateMonitor callback
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # Set up the PyTorch Lightning Trainer 
        self.trainer = pl.Trainer(
            num_sanity_val_steps=1,  # Number of validation sanity steps
            devices=[0],  # List of GPU devices to use
            max_epochs=self.epochs,  # Maximum number of epochs to train
            logger=self.logger,  # Logger for experiment tracking
            accelerator="cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available, otherwise CPU
            callbacks=[checkpoint_callback, val_rollout_callback, lr_monitor],  # List of callbacks to use during training
        )
        
    def _init_net(self):
        """
        Initializes the neural network model.
        
        Args:
            args (object): Command line arguments or configuration settings.
        """
        print("Initializing the model...")
        self.model = TIGNN(self.args, self.stats, self.noise_var, lambda_d=self.lambda_d)
        print("Model initialized.")
    
    def _init_pretrained_net(self):
        """
        Initializes the pretrained neural network model.
        
        Args:
            args (object): Command line arguments or configuration settings.
        """
        print("Initializing the pretrained model...")
        # Define the path to the best checkpoint file
        checkpoint_path = 'outputs/saved_models/best_inference/best_model.ckpt'

        try:
            # Load the best checkpoint weights using the PINN class
            print(f"Loading weights from {checkpoint_path}")
            self.model = TIGNN.load_from_checkpoint(checkpoint_path, args = self.args, stats = self.stats, noise = self.noise_var, lambda_d=self.lambda_d)
            print("Pretrained model initialized.")
        except FileNotFoundError:
            print(f"Checkpoint file not found at {checkpoint_path}. Testing with current model weights.")


    def train(self):
        # Train the model using the Trainer
        self.trainer.fit(self.model, self.train_loader, self.valid_loader)

    def test(self):
        """
        Tests the model using the Trainer and logs the results.
        """
        try:
            # Initialize the pretrained network
            self._init_pretrained_net()            
            # Test the model using the Trainer
            self.trainer.test(self.model, self.test_loader)           
            # Calculate the mean degeneration on inference
            mean_degeneration = sum(self.model.deg_list) / len(self.model.deg_list)
            print(f"Mean Degeneration on inference: {mean_degeneration}")           
            # Log the mean degeneration to Wandb
            wandb.log({"mean_degeneration_test": mean_degeneration})           
            # Prepare test results for plotting
            test_results = {'z_net': self.model.z_net_list, 'z_gt': self.model.z_gt_list}           
            # Create plots for the test results
            create_plots(test_results['z_net'], test_results['z_gt'], self.model.n_list, self.model.w_edge_list, 'Test/Test_Unseen')           
            # Plot rollout error if specified
            if self.plot_rollout:
                plot_rollout_error(self.args, test_results, self.model.n_list, node=0, sim=0, noise=self.noise_var, u_mean=self.stats['stats_u_m']['means'])
            # Evaluate and print the error
            error, _ = evaluate_and_print(test_results, set_name='Test_Unseen')
            # Create a boxplot of the error
            boxplot_error(error, name='Test/Test_Unseen')
            # Plot 2D CA of the error
            plot_2D_ca(error, self.model.total_loads, self.model.total_f_u)
        
        except Exception as e:
            # Handle any exceptions that occur during testing
            print(f"Testing failed: {e}")
            return