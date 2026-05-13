import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        out = []
        with torch.no_grad():
            for layer in model.children():
                x = layer(x) # manually perform calc
                if isinstance(layer, nn.Linear): # record diagnostics
                    mean = x.mean().round(decimals=4).item()
                    std = x.std().round(decimals=4).item()    
                    dead_frac = ((x <= 0).all(dim=0).float().mean()).round(decimals=4).item()
                    out.append({"mean" : mean, "std": std, "dead_fraction" : dead_frac})
        return out
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        pass

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        out = []
        
        model.zero_grad()
        # generate prediction
        y_hat = model(x)
        criterion = nn.MSELoss()
        loss = criterion(y_hat, y) # check against actual for error
        loss.backward()
    
        for layer in model.children():
            if isinstance(layer, nn.Linear): # record diagnostics
                mean = layer.weight.grad.mean().round(decimals=4).item()
                std = layer.weight.grad.std().round(decimals=4).item()
                norm = layer.weight.grad.norm().round(decimals=4).item()
                
                out.append({"mean" : mean, "std" : std, "norm" : norm})
            
        return out

        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        pass

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        
        for layer in activation_stats:
            if layer["dead_fraction"] > 0.5:
                return "dead_neurons"
            if layer["std"] < 0.1 :
                return "vanishing_gradients"
            elif layer["std"] > 10.0:
                return "exploding_gradients"

        for layer in gradient_stats:
            if layer["norm"] > 1000:
                return "exploding_gradients"
            # if last layer gradient < 1e-5
                # return "vanishing_gradients"
        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        return "healthy"
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        pass
