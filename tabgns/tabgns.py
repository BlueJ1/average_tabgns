import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch import Tensor

from typing import List

from tabgns.gates import GatedBlock
from tabgns.mlp import MLP
from tabgns.utils import param_count, EarlyStopper, accuracy


class GatedNet(nn.Module):
  def __init__(self, config: dict):
    super().__init__()
    self.config = config
    self.d_in, self.d_out = self.config['general']['inp_dim'], self.config['general']['out_dim']
    self.gate_params: List[Tensor] = []
    self.weight_params: List[nn.Parameter] = []
    self._criterion = nn.MSELoss() if config['general']['criterion'] == 'mse' else nn.CrossEntropyLoss()

    if 'layer_widths' in self.config['search_space']:
      layer_widths = self.config['search_space']['layer_widths']

    inp =  self.d_in
    blocks: List[GatedBlock] = []
    for i, out in enumerate(layer_widths):
      block = GatedBlock(
        in_features=inp,
        out_features=out,
        gate_initial_value=self.config['tabgns']['gate_initialisation_val'],
        temperature=self.config['tabgns']['temperature']
      )
      blocks.append(block)
      inp = out

      self.gate_params.extend([block.arch_parameters()])
      self.weight_params.extend(block.weight_parameters())

    self.blocks = nn.ModuleList(blocks)
    self.output_layer = nn.Linear(layer_widths[-1], self.config['general']['out_dim'])
    self.weight_params.extend(self.output_layer.parameters())
    
  def forward(self, x: Tensor) -> Tensor:
    for block in self.blocks:
        x = block(x)
    x = self.output_layer(x)
    return x

  def new(self) -> "GatedNet":
    model_new = GatedNet(self.config)
    with torch.no_grad():
      for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
      for x, y in zip(model_new.weight_parameters(), self.weight_parameters()):
        x.data.copy_(y.data)
    return model_new

  def mask(self) -> List[np.ndarray]:
    """Binarised gate values"""
    gates = []
    for block in self.blocks:
      gates.append(torch.round(block.sigmoid_values().cpu()).detach().numpy())
    return gates

  def gate_values(self) -> List[Tensor]:
    return [block.gate_values() for block in self.blocks]
    
  def continuous_params_num(self) -> Tensor:
    vals = self.gate_values()
    layers = [torch.sum(l) for l in vals]
    out = torch.tensor(0., requires_grad=True)
    for i in range(len(layers)-1):
        out = out + (layers[i] * layers[i+1]) 
    return out

  def curr_architecture(self) -> List[int]:
    return [int(sum(gates)) for gates in self.mask()]

  def arch_parameters(self) -> List[Tensor]:
    return self.gate_params

  def weight_parameters(self) -> List[nn.Parameter]:
    return self.weight_params

  def named_weight_parameters(self) -> List[tuple]:
    named_weight_params = []
    for name, param in self.named_parameters():
      is_gate_param = False
      if not 'strength' in name:
        named_weight_params.append((name, param))
    return named_weight_params

  def _loss(self, input: Tensor, target: Tensor, grad: bool=True) -> Tensor:
    logits = self.forward(input)
    loss = self._criterion(logits, target)
    return loss

class TabGNS:
  def __init__(self, config: dict) -> None:
    self.arch_hist = []
    self.config = config
    self.net = GatedNet(config)
    self.early_stopper = EarlyStopper(patience=config['general']['stop_patience'], model=self.net)
    
    self.weight_optimiser = optim.Adam(self.net.weight_parameters(),
      lr=config['general']['weight_lr'], weight_decay=0.)
    self.arch_optimiser = optim.Adam(self.net.arch_parameters(),
      lr=config['tabgns']['arch_lr'], weight_decay=0.)
    
  def arch_parameters(self) -> List[nn.Parameter]:
    return self.net.arch_parameters()
    
  def weight_parameters(self) -> List[nn.Parameter]:
    return self.net.weight_parameters()

  def fit(self, train_loader: DataLoader, valid_loader: DataLoader):
    pbar = tqdm(range(self.config['general']['iters']))
    self.net = self.net.to(self.config['general']['device'])
    
    valid_iter = iter(valid_loader)
    
    for epoch in pbar:
      train_loss, valid_loss, valid_criterion = 0., 0., 0.
      val_acc = 0.
      
      for step, (input, target) in enumerate(train_loader):
        self.net.train()
        n = input.size(0)

        input = input.to(self.config['general']['device'])
        target = target.to(self.config['general']['device'])
        try:
          input_search, target_search = next(valid_iter)
        except StopIteration:
          valid_iter = iter(valid_loader)
          input_search, target_search = next(valid_iter)
        input_search = input_search.to(self.config['general']['device'])
        target_search = target_search.to(self.config['general']['device'])
        
        if isinstance(self.net._criterion, nn.CrossEntropyLoss):
          if target.ndim > 1 and target.size(-1) > 1:
            target = torch.argmax(target, dim=1).long()
          elif target.ndim > 1 and target.size(-1) == 1:
            target = target.squeeze(-1).long()
          else:
            target = target.long()
          if target_search.ndim > 1 and target_search.size(-1) > 1:
            target_search_indices = torch.argmax(target_search, dim=1).long()
          elif target_search.ndim > 1 and target_search.size(-1) == 1:
            target_search_indices = target_search.squeeze(-1).long()
          else:
            target_search_indices = target_search.long()
        else:
          target_search_indices = target_search
          
        self.weight_optimiser.zero_grad()
        logits = self.net(input)
        loss = self.net._criterion(logits, target)
        train_loss += loss.item()
        loss.backward()
        self.weight_optimiser.step()
        
        with torch.no_grad():
          self.net.eval()
          valid_loss += self.net._loss(input_search, target_search_indices, grad=False).item()
          val_acc += accuracy(target_search, self.net(input_search))

        self.arch_optimiser.zero_grad()
        logits = self.net(input_search)
        loss = self.net._criterion(logits, target_search_indices)
        loss.backward()
        self.arch_optimiser.step()
        
      n_params = param_count(self.net.curr_architecture())
      self.arch_hist.append(self.net.curr_architecture())
      train_loss /= len(train_loader)
      valid_loss /= len(train_loader)
      val_acc /= len(train_loader)

      pbar.set_description(f"Epochs: {epoch} Train Loss {train_loss:.4f}, Val Acc {val_acc:.4f}, N params {n_params}")

      if self.early_stopper(valid_loss, self.net):
        break
  
    self.net = self.early_stopper.best_model
    return self.extract_net()

  def curr_architecture(self) -> List[int]:
    return self.net.curr_architecture()
    
  def get_architecture(self) -> List[int]:
    return self.curr_architecture()

  def extract_net(self) -> MLP:
    self.net.to('cpu')

    layer_widths = self.curr_architecture()
    mlp_config = {**self.config}
    mlp_config.update({'layer_widths': layer_widths})
    subnet = MLP(mlp_config)

    cols_indexes = torch.arange(int(self.net.blocks[0].linear.in_features))
    src = list(self.net.blocks)
    trg = [m for m in subnet.net if isinstance(m, torch.nn.Linear)]
    for src_block, trg_block in zip(src, trg[:-1]):
      gate_vals = src_block.sigmoid_values()
      on_gates = torch.where(gate_vals >= 0.5,
        torch.ones_like(gate_vals),
        torch.zeros_like(gate_vals)
      )

      rows_indexes = torch.nonzero(on_gates).flatten()

      weights = src_block.linear.weight
      bias = src_block.linear.bias

      copy_gates = torch.index_select(gate_vals, 0, rows_indexes)
      copy_weights = torch.index_select(weights, 0, rows_indexes)
      copy_bias = torch.index_select(bias, 0, rows_indexes)

      copy_weights = torch.index_select(copy_weights, 1, cols_indexes)
      cols_indexes = rows_indexes.clone().detach()

      if self.config['tabgns']['scale']:
        copy_weights = copy_weights * copy_gates.unsqueeze(1)
        copy_bias = copy_bias * copy_gates

      with torch.no_grad():
        trg_block.weight.data = copy_weights
        trg_block.bias.data = copy_bias

    copy_weights = self.net.output_layer.weight.clone().detach()
    copy_weights = torch.index_select(copy_weights, 1, cols_indexes)
    copy_bias = self.net.output_layer.bias.clone().detach()
    with torch.no_grad():
      trg[-1].weight.data = copy_weights
      trg[-1].bias.data = copy_bias
    return subnet

  def predict(self, x: Tensor, using: str = 'supernet') -> Tensor:
    assert using in ['supernet', 'subnet']
    if using == 'supernet':
      x = x.to(self.config['general']['device'])
      return self.net(x).cpu()
    else:
      raise NotImplementedError()
