import torch

from watchmal.engine.reconstruction import ReconstructionEngine

from torch.amp import autocast

import numpy as np




class ClassifierEngine(ReconstructionEngine):
    """Engine for performing training or evaluation for a classification network."""
    def __init__(self, truth_key, model, rank, gpu, dump_path,
                 label_set=None, eval_directory='/', mixup_alpha=0.0):
        """
        Parameters
        ==========
        truth_key : string
            Name of the key for the target labels in the dictionary returned by the dataloader
        model
            `nn.module` object that contains the full network that the engine will use in training or evaluation.
        rank : int
            The rank of process among all spawned processes (in multiprocessing mode).
        gpu : int
            The gpu that this process is running on.
        dump_path : string
            The path to store outputs in.
        label_set : sequence
            The set of possible labels to classify (if None, which is the default, then class labels in the data must be
            0 to N).
        mixup_alpha : float
            The alpha parameter for the mixup data augmentation. 0.0 disables mixup.
        """
        # create the directory for saving the log and dump files
        super().__init__(truth_key, model, rank, gpu, dump_path)
        self.softmax = torch.nn.Softmax(dim=1)
        self.eval_directory = eval_directory
        self.label_set = label_set
        self.mixup_alpha = mixup_alpha  # 0.0 disables mixup

    def configure_data_loaders(self, data_config, loaders_config, is_distributed, seed):
        """
        Set up data loaders from loaders hydra configs for the data config, and a list of data loader configs.

        Parameters
        ==========
        data_config
            Hydra config specifying dataset.
        loaders_config
            Hydra config specifying a list of dataloaders.
        is_distributed : bool
            Whether running in multiprocessing mode.
        seed : int
            Random seed to use to initialize dataloaders.
        """
        super().configure_data_loaders(data_config, loaders_config, is_distributed, seed)
        if self.label_set is not None:
            for name in loaders_config.keys():
                self.data_loaders[name].dataset.map_labels(self.label_set)

    def forward(self, train=True):
        """
        Compute predictions and metrics for a batch of data.

        Parameters
        ==========
        train : bool
            Whether in training mode, requiring computing gradients for backpropagation

        Returns
        =======
        dict
            Dictionary containing loss, predicted labels, softmax, accuracy, and raw model outputs
        """
        with torch.set_grad_enabled(train):
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                
                if train and self.mixup_alpha > 0.0:
                    # Sample lambda from Beta(alpha, alpha)
                    lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    batch_size = self.data.size(0)
                    perm = torch.randperm(batch_size, device=self.device)

                    mixed_data = lam * self.data + (1 - lam) * self.data[perm]
                    labels_a = self.target
                    labels_b = self.target[perm]

                    model_out = self.model(mixed_data)
                    self.loss = lam * self.criterion(model_out, labels_a) \
                              + (1 - lam) * self.criterion(model_out, labels_b)
                    # Accuracy is measured against the dominant label
                    predicted_labels = torch.argmax(model_out, dim=-1)
                    accuracy = (predicted_labels == labels_a).sum() / float(predicted_labels.nelement())
                else:
                    model_out = self.model(self.data)
                    self.loss = self.criterion(model_out, self.target)
                    predicted_labels = torch.argmax(model_out, dim=-1)
                    accuracy = (predicted_labels == self.target).sum() / float(predicted_labels.nelement())

                softmax = self.softmax(model_out)

        outputs = {'softmax': softmax.detach()}
        metrics = {'loss': self.loss, 'accuracy': accuracy}
        return outputs, metrics
