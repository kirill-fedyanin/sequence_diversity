import torch
from torch import nn, Tensor
import types

class UeDropout(nn.Dropout):
    def forward_share_across_tokens(self, input: Tensor) -> Tensor:
        shape = input.shape
        mask_shape = shape[2:]
        mask = torch.empty(mask_shape, dtype=torch.bool, device=input.device).bernoulli_(
                           self.p)
        mask = mask.repeat(list(shape[:2]) + [1] * len(mask_shape))
        return input.masked_fill(mask, 0) / (1 - self.p)

    def forward(self, input: Tensor) -> Tensor:
        shape = input.shape
        mask_shape = shape[1:]
        mask = torch.empty(mask_shape, dtype=torch.bool, device=input.device).bernoulli_(
                           self.p)
        mask = mask.repeat(list(shape[:1]) + [1] * len(mask_shape))
        return input.masked_fill(mask, 0) / (1 - self.p)

    def indentity(self, input: Tensor) -> Tensor:
        return input

def replace_with_identity(module):
      children = module.children()
      for c_mod in children:
          if type(c_mod).__name__ == "Dropout":
              method = getattr(UeDropout, 'identity')
              setattr(c_mod, 'forward', types.MethodType(method, mod))
          else:
              replace_with_identity(c_mod)

def replace_dropout(module, p=0.1, share_across_tokens=True):
      children = module.children()
      for c_mod in children:
          #if type(c_mod).__name__ in ["T5LayerSelfAttention", "T5LayerCrossAttention"]:
          #    replace_with_identity(c_mod)
          if type(c_mod).__name__ == "Dropout":
              if share_across_tokens:
                  method = getattr(UeDropout, 'forward_share_across_tokens')
              else:
                  method = getattr(UeDropout, 'forward')
              setattr(c_mod, 'forward', types.MethodType(method, c_mod))
              c_mod.p = p
          else:
              replace_dropout(c_mod, p=p, share_across_tokens=share_across_tokens)
