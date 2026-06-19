import torch
import inspect

from watchmal.engine.reconstruction import ReconstructionEngine

import analysis.utils.math as math

from torch.amp import autocast


class RegressionEngine(ReconstructionEngine):
    """Engine for performing training or evaluation for a regression network."""
    def __init__(self, truth_key, model, rank, gpu, dump_path, output_center=0, output_scale=1, eval_directory='/', truth_key_size=[1]):
        """
        Parameters
        ==========
        truth_key : string
            Name of the key for the target values in the dictionary returned by the dataloader
        model
            `nn.module` object that contains the full network that the engine will use in training or evaluation.
        rank : int
            The rank of process among all spawned processes (in multiprocessing mode).
        gpu : int
            The gpu that this process is running on.
        dump_path : string
            The path to store outputs in.
        output_center : float
            Value to subtract from target values
        output_scale : float
            Value to divide target values by
        """
        # create the directory for saving the log and dump files
        super().__init__(truth_key, model, rank, gpu, dump_path, truth_key_size=truth_key_size)
        self.output_center = output_center
        self.output_scale = output_scale
        self.eval_directory=eval_directory

    def forward(self, train=True):
        """
        Compute predictions and metrics for a batch of data

        Parameters
        ==========
        train : bool
            Whether in training mode, requiring computing gradients for backpropagation

        Returns
        =======
        dict
            Dictionary containing loss and predicted values
        """
        print("0")
        with torch.set_grad_enabled(train):
            print("1")
            # 1. Wrap operations in native bfloat16 autocast to save VRAM and boost throughput
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                print("2")
                model_out = self.model(self.data).reshape(self.target.shape)
                print("3")
                if self.rank == 0:
                    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    print(f"GPU memory reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                print("4")
                
                scaled_target = self.scale_values(self.target)
                scaled_model_out = self.scale_values(model_out)
                
                if 'energy' in inspect.signature(self.criterion.forward).parameters:
                    self.loss = self.criterion(scaled_model_out, scaled_target, energy=self.energy)
                else:
                    self.loss = self.criterion(scaled_model_out, scaled_target)
                
                if self.dir is not None and train is False:
                    longitudinal_component_pred = math.decompose_along_direction_pytorch(
                        scaled_model_out[:, 0:3] - scaled_target[:, 0:3], self.dir
                    )

            # 2. Build outputs dictionary (detach to prevent graph trailing)
            if self.multi_key:
                base = 0
                outputs = {}
                for i, key in enumerate(self.truth_key):
                    outputs["predicted_" + str(key)] = model_out[:, base:base + self.truth_key_size[i]].detach()
                    base = base + self.truth_key_size[i]
            else:
                outputs = {"predicted_" + self.truth_key: model_out.detach()}
            
                        # 3. CRITICAL: Detach the loss tensor into a pure Python scalar (.item()) 
            # This isolates metrics from the live VRAM computation graph.
            if False and (self.dir is not None and train is False):
                (numerical_bot_quantile, numerical_top_quantile) = torch.quantile(
                    longitudinal_component_pred, torch.tensor([0.159, 0.841]).to(self.device)
                )
                numerical_median = torch.median(longitudinal_component_pred)
                quantile = (torch.abs((numerical_median - numerical_bot_quantile)) + torch.abs((numerical_median - numerical_top_quantile))) / 2
                metrics = {'loss': self.loss.item(), 'long_res': quantile.item()}
            else:
                metrics = {'loss': self.loss.item()} # 👈 Fixed leak (.item() removes the graph)

        return outputs, metrics

    def scale_values(self, data):
        scaled = (data - self.output_center) / self.output_scale
        return scaled
