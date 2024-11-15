from rob831.hw4_part2.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # <TODO>: Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f

        self.f = ptu.build_mlp(input_dim=self.ob_dim, output_dim=self.output_size, 
                            num_layers=self.n_layers, layer_size=self.size, 
                            init_method=init_method_1)
        self.f_hat = ptu.build_mlp(input_dim=self.ob_dim, output_dim=self.output_size, 
                                num_layers=self.n_layers, layer_size=self.size, 
                                init_method=init_method_2)

        # Set up the optimizer for f_hat
        self.optimizer = self.optimizer_spec.constructor(self.f_hat.parameters(), **self.optimizer_spec.optim_kwargs)

        # Initialize the learning rate scheduler
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, 
                                                                lr_lambda=self.optimizer_spec.learning_rate_schedule)

        # Move models to the correct device (e.g., GPU or CPU)
        self.f.to(ptu.device)
        self.f_hat.to(ptu.device)


    def forward(self, ob_no):
        # <TODO>: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        return torch.norm((self.f_hat(ob_no)-self.f(ob_no).detach()), dim=1)

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # <TODO>: Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        self.optimizer.zero_grad()
        loss = torch.mean(self(ptu.from_numpy(ob_no)))
        loss.backward()
        self.optimizer.step()
        return loss.item()
