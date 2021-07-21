import time
from pathlib import Path
import numpy as np
import torch
import os
from util import summary
from util.utils import prepare_empty_dir, ExecutionTime, prepare_device

class BaseTrainer:
    def __init__(self, resume: bool, model, loss_function, optimizer, scheduler, epochs, save_checkpoint_interval, test_interval, find_max):
        self.n_gpu = torch.cuda.device_count()
        self.device = prepare_device(self.n_gpu)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function

        self.model = model.to(self.device)

        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.n_gpu)))

        # Trainer
        self.epochs = epochs
        self.save_checkpoint_interval = save_checkpoint_interval
        self.find_max = find_max
        # The following args is not in the config file, We will update it if resume is True in later.
        self.start_epoch = 1
        self.test_interval = test_interval
        self.best_score = -np.inf if self.find_max else np.inf
        self.root_dir = './output/'
        self.checkpoints_dir = os.path.join(self.root_dir, "checkpoints")
        self.logs_dir = os.path.join(self.root_dir, "logs")
        prepare_empty_dir([self.checkpoints_dir, self.logs_dir], resume=resume)
        self.writer = summary.writer(self.logs_dir)
        '''
        self.writer.add_text(
            tag="Configuration",
            text_string=f"<pre>  \n{json5.dumps(config, indent=4, sort_keys=False)}  \n</pre>",
            global_step=1
        )
        '''
        self.global_step = 1
        if resume: self._resume_checkpoint()
        '''
        if config["preloaded_model_path"]: self._preload_model(Path(config["preloaded_model_path"]))

        print("Configurations are as follows: ")
        print(json5.dumps(config, indent=2, sort_keys=False))

        with open((self.root_dir / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}.json"), "w") as handle:
            json5.dump(config, handle, indent=2, sort_keys=False)
        '''
        self._print_networks([self.model])

    def _preload_model(self, model_path):
        """
        Preload *.pth file of the model at the start of the current experiment.

        Args:
            model_path(Path): the path of the *.pth file
        """
        model_path = model_path.expanduser().absolute()
        assert model_path.exists(), f"Preloaded *.pth file is not exist. Please check the file path: {model_path}"
        model_checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(model_checkpoint, strict=False)
        else:
            self.model.load_state_dict(model_checkpoint, strict=False)

        print(f"Model preloaded successfully from {model_path}.")

    def _resume_checkpoint(self):
        """Resume experiment from latest checkpoint.
        Notes:
            To be careful at Loading model. if model is an instance of DataParallel, we need to set model.module.*
        """
        latest_model_path = self.checkpoints_dir.expanduser().absolute() / "latest_model.tar"
        assert latest_model_path.exists(), f"{latest_model_path} does not exist, can not load latest checkpoint."

        checkpoint = torch.load(latest_model_path, map_location=self.device)

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])

        print(f"Model checkpoint loaded. Training will begin in {self.start_epoch} epoch.")

    def _save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint to <root_dir>/checkpoints directory, which contains:
            - current epoch
            - best score in history
            - optimizer parameters
            - model parameters
        Args:
            is_best(bool): if current checkpoint got the best score, it also will be saved in <root_dir>/checkpoints/best_model.tar.
        """
        print(f"\t Saving {epoch} epoch model checkpoint...")

        # Construct checkpoint tar package
        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "optimizer": self.optimizer.state_dict()
        }

        if isinstance(self.model, torch.nn.DataParallel):  # Parallel
            state_dict["model"] = self.model.module.cpu().state_dict()
        else:
            state_dict["model"] = self.model.cpu().state_dict()

        """
        Notes:
            - latest_model.tar:
                Contains all checkpoint information, including optimizer parameters, model parameters, etc. New checkpoint will overwrite old one.
            - model_<epoch>.pth: 
                The parameters of the model. Follow-up we can specify epoch to inference.
            - best_model.tar:
                Like latest_model, but only saved when <is_best> is True.
        """
        torch.save(state_dict, os.path.join(self.checkpoints_dir, "latest_model.tar"))
        torch.save(state_dict["model"], os.path.join(self.checkpoints_dir, f"model_{str(epoch).zfill(4)}.pth"))
        if is_best:
            print(f"\t Found best score in {epoch} epoch, saving...")
            torch.save(state_dict, os.path.join(self.checkpoints_dir, "best_model.tar"))

        # Use model.cpu() or model.to("cpu") will migrate the model to CPU, at which point we need re-migrate model back.
        # No matter tensor.cuda() or tensor.to("cuda"), if tensor in CPU, the tensor will not be migrated to GPU, but the model will.
        self.model.to(self.device)

    def _is_best(self, score, find_max=True):
        """Check if the current model is the best model
        """
        if find_max and score >= self.best_score:
            self.best_score = score
            return True
        elif not find_max and score <= self.best_score:
            self.best_score = score
            return True
        else:
            return False

    @staticmethod
    def _print_networks(nets: list):
        print(f"This project contains {len(nets)} networks, the number of the parameters: ")
        params_of_all_networks = 0
        for i, net in enumerate(nets, start=1):
            params_of_network = 0
            for param in net.parameters():
                params_of_network += param.numel()

            print(f"\tNetwork {i}: {params_of_network / 1e6} million.")
            params_of_all_networks += params_of_network

        print(f"The amount of parameters in the project is {params_of_all_networks / 1e6} million.")

    def _set_models_to_train_mode(self):
        self.model.train()

    def _set_models_to_eval_mode(self):
        self.model.eval()

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"============== {epoch} epoch ==============")
            print("[0 seconds] Begin training...")
            timer = ExecutionTime()

            self._set_models_to_train_mode()
            self._train_epoch(epoch)

            print(f"[{timer.duration()} seconds] End this epoch.")

            if self.save_checkpoint_interval != 0 and (epoch % self.save_checkpoint_interval == 0):
                self._save_checkpoint(epoch)

            if self.test_interval != 0 and epoch % self.test_interval == 0:
                print(f"[{timer.duration()} seconds] Training is over, Testing is in progress...")

                self._set_models_to_eval_mode()
                print("[0 seconds] Begin testing...")
                score = self._test_epoch(epoch)

                if self._is_best(score, find_max=self.find_max):
                    self._save_checkpoint(epoch, is_best=True)

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _test_epoch(self, epoch):
        raise NotImplementedError

