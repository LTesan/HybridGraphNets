import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from src.utils.compute_error import evaluate_and_print
import wandb
from src.utils.plots import create_plots, boxplot_error

class ValidationRolloutCallback(Callback):
    def __init__(self, dataloader, interval=50, save_dir='outputs/saved_models/best_inference'):
        super().__init__()
        self.interval = interval
        self.val_epoch_counter = 0
        self.dataloader = dataloader
        self.best_mean_error = float('inf')  # Initialize best error to a high value
        wandb.log({'best_mean_error': self.best_mean_error})
        self.save_dir = save_dir

        # Ensure the save directory exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def on_validation_epoch_end(self, trainer, pl_module):

        # Ensure that the epoch count is greater than interval before running the validation
        self.val_epoch_counter += 1
        # Check if it's time to run the additional validation logic
        if self.val_epoch_counter % self.interval == 0:
            print('\nRunning validation for rollout inference...\n')

            # Initialize lists to store validation results
            z_net_list_val = []
            z_gt_list_val = []
            n_list = []
            w_edge_list = []

            # Set the model to evaluation mode to disable gradient computations
            pl_module.eval()

            # Run test on validation data, limiting to max_batches
            for i, batch in enumerate(self.dataloader):
                z_net, z_gt, w_edge, n = pl_module.integrate_sim(batch, full_rollout=True)
                z_net_list_val.append(z_net)
                z_gt_list_val.append(z_gt)
                w_edge_list.append(w_edge)
                n_list.append(n)

            # Collect and evaluate validation results
            val_results = {'z_net': z_net_list_val, 'z_gt': z_gt_list_val}
            create_plots(z_net_list_val, z_gt_list_val, n_list, w_edge_list, 'Val/Val_Unseen')
            error, mean_error = evaluate_and_print(val_results, set_name='Val_Unseen', save=False)
            boxplot_error(error, name=f'Val/Val_Unseen')

            # Check if the current mean error is the best one so far
            if mean_error < self.best_mean_error:
                print(f'New best mean error {mean_error:.4f} (previous best was {self.best_mean_error:.4f}). Saving model...')
                self.best_mean_error = mean_error
                wandb.log({'best_mean_error': self.best_mean_error})

                # Save the full checkpoint using PyTorch Lightning's Trainer
                model_save_path = os.path.join(self.save_dir, f'best_model.ckpt')
                trainer.save_checkpoint(model_save_path)
                print(f'Model saved to {model_save_path}')

            # Reset the validation epoch counter
            if self.val_epoch_counter % self.interval == 0:
                self.val_epoch_counter = 0
