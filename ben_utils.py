#!/usr/bin/env python3
import torch as t

class SaveForward:
    def __init__(self, layer):
        self.layer = layer

    def __enter__(self):
        self.remove_handle = self.layer.register_forward_hook(self.hook)
        return self

    def __exit__(self, exn_ty, exn_val, exn_tb):
        self.remove_handle.remove()

    def hook(self, module, module_input, module_output):
        self.saved_input = module_input
        self.saved_output = module_output
