import torch
import torch.nn as nn
from utils import _get_act, _get_initializer
class FNN(nn.Module):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        self.activation = _get_act(activation)
        initializer = _get_initializer(kernel_initializer)
        initializer_zero = _get_initializer("zeros")

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=torch.float32
                )
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

    def forward(self, inputs):
        x = inputs
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        return x
    
class DeepONet(nn.Module):
    """Deep operator network.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = _get_act(activation["branch"])
            self.activation_trunk = _get_act(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = _get_act(activation)
        if callable(layer_sizes_branch[0]):
            # User-defined network
            self.branch = layer_sizes_branch[0]
        else:
            # Fully connected network
            self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        x_loc = self.activation_trunk(self.trunk(x_loc))
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum("bi,bi->b", x_func, x_loc)
        x = torch.unsqueeze(x, 1)
        # Add bias
        x += self.b
        return x

class DeepONetCartesianProd(nn.Module):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = _get_act(activation["branch"])
            self.activation_trunk = _get_act(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = _get_act(activation)
        if callable(layer_sizes_branch[0]):
            # User-defined network
            self.branch = layer_sizes_branch[0]
        else:
            # Fully connected network
            self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        x_loc = self.activation_trunk(self.trunk(x_loc))
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum("bi,ni->bn", x_func, x_loc)
        # Add bias
        x += self.b
        return x

class DeepONetCartesianProd2D(DeepONetCartesianProd):
    #   Apply multi-layer CNN with a global mean pooling for the branch net to handle 2d rectangle input function. 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        in_channel_branch: int,
        query_dim: int ,
        out_channel: int,
        activation: str = "relu",
        kernel_initializer: str = "Glorot normal"):

        layer_sizes_trunk= [query_dim]+[128]*3+[128*out_channel]
        branchnet=nn.Sequential(nn.Conv2d(in_channel_branch,64,kernel_size=5,stride=2),
                      nn.GELU(),
                      nn.Conv2d(64,128,kernel_size=5,stride=2),
                      nn.GELU(),
                      nn.AdaptiveAvgPool2d((1,1)),
                      nn.Flatten(),
                      nn.Linear(128,128),
                      nn.GELU(),
                      nn.Linear(128,128*out_channel),
                      )
        super().__init__([branchnet],layer_sizes_trunk,activation,kernel_initializer)
        self.out_channel = out_channel
        self.query_dim=query_dim
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]  
        grid_shape = x_loc.shape[:-1]
        x_loc= x_loc.reshape([-1,self.query_dim])  #(num_point, query_dim)
        batchsize=x_func.shape[0]
        num_points=x_loc.shape[0] 
        # Branch net to encode the input function
        # breakpoint()
        x_func = self.branch(x_func.permute((0, 3, 1, 2)))
        # Trunk net to encode the domain of the output function
        x_loc = self.activation_trunk(self.trunk(x_loc))
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x_func = x_func.reshape([batchsize,self.out_channel,-1])
        x_loc = x_loc.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x_func, x_loc)
        # Add bias
        x += self.b
        return x.reshape([-1,*grid_shape,self.out_channel]).unsqueeze(-2)

# class DeepONetCartesianProd1D(DeepONetCartesianProd):
#     #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
#     # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
#     def __init__(self,
#         in_channel_branch: int,
#         in_length_branch: int,
#         query_dim: int ,
#         out_channel: int,
#         activation: str = "relu",
#         kernel_initializer: str = "Glorot normal"):

#         layer_sizes_trunk= [query_dim]+[128]*3+[128*out_channel]
#         layer_sizes_branch= [in_channel_branch*in_length_branch]+[128]*3+[128*out_channel]
#         super().__init__(layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer)
#         self.out_channel = out_channel
#         self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

#     def forward(self, inputs):
#         x_func = inputs[0]
#         x_loc = inputs[1]
#         batchsize=x_func.shape[0]
#         num_points=x_loc.shape[0]
#         # Branch net to encode the input function
#         x_func = self.branch(x_func.reshape([batchsize,-1]))
#         # Trunk net to encode the domain of the output function
#         x_loc = self.activation_trunk(self.trunk(x_loc))
#         # Dot product
#         if x_func.shape[-1] != x_loc.shape[-1]:
#             raise AssertionError(
#                 "Output sizes of branch net and trunk net do not match."
#             )
#         x_func = x_func.reshape([batchsize,self.out_channel,-1])
#         x_loc = x_func.reshape([num_points,self.out_channel,-1])
#         x = torch.einsum("bci,nci->bnc", x_func, x_loc)
#         # Add bias
#         x += self.b
#         return x

class DeepONetCartesianProd1D(DeepONetCartesianProd):
    #   Apply multi-layer CNN with a global mean pooling for the branch net to handle 2d rectangle input function. 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        in_channel_branch: int,
        query_dim: int ,
        out_channel: int,
        activation: str = "relu",
        kernel_initializer: str = "Glorot normal"):

        layer_sizes_trunk= [query_dim]+[128]*3+[128*out_channel]
        branchnet=nn.Sequential(nn.Conv1d(in_channel_branch,64,kernel_size=5,stride=2),
                      nn.GELU(),
                      nn.Conv1d(64,128,kernel_size=5,stride=2),
                      nn.GELU(),
                      nn.AdaptiveAvgPool1d((1,)),
                      nn.Flatten(),
                      nn.Linear(128,128),
                      nn.GELU(),
                      nn.Linear(128,128*out_channel),
                      )
        super().__init__([branchnet],layer_sizes_trunk,activation,kernel_initializer)
        self.out_channel = out_channel
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))
        self.query_dim=query_dim

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        grid_shape = x_loc.shape[:-1]
        x_loc= x_loc.reshape([-1,self.query_dim])  #(num_point, query_dim)
        batchsize=x_func.shape[0]
        num_points=x_loc.shape[0]
        # Branch net to encode the input function
        
        x_func = x_func.permute((0, 2, 1))
        # breakpoint()
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        x_loc = self.activation_trunk(self.trunk(x_loc))
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x_func = x_func.reshape([batchsize,self.out_channel,-1])
        x_loc = x_loc.reshape([num_points,self.out_channel,-1])
        # breakpoint()
        x = torch.einsum("bci,nci->bnc", x_func, x_loc)
        # Add bias
        x += self.b
        return x.reshape([-1,*grid_shape,self.out_channel]).unsqueeze(-2)