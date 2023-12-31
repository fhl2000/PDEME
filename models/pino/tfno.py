import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch    # conda install -c conda-forge tensorly tensorly-torch



@torch.jit.script
def contract_1D(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bix,iox->box", a, b)
    return res


@torch.jit.script
def contract_2D(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixy,ioxy->boxy", a, b)
    return res


@torch.jit.script
def contract_3D(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixyz,ioxyz->boxyz", a, b)
    return res


class FactorizedSpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_height, modes_width, modes_depth, n_layers=1, bias=True, scale='auto',
                 fft_norm='backward', mlp=False,
                 rank=0.5, factorization='cp', fixed_rank_modes=None, decomposition_kwargs=dict(), **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_height = modes_height
        self.modes_width = modes_width
        self.modes_depth = modes_depth
        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers
        self.fft_norm = fft_norm
        if mlp:
            raise NotImplementedError()
        else:
            self.mlp = None

        if scale == 'auto':
            scale = (1 / (in_channels * out_channels))

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                fixed_rank_modes=[0]
            else:
                fixed_rank_modes=None

        if factorization is None:
            self.weight = nn.Parameter(scale * torch.randn(4*n_layers, in_channels, out_channels, self.modes_height, self.modes_width, self.modes_depth,
                                                            dtype=torch.cfloat))
            self._get_weight = self._get_weight_dense
        else:
            self.weight = tltorch.FactorizedTensor.new((4*n_layers, in_channels, out_channels, self.modes_height, self.modes_width, self.modes_depth),
                                                        rank=self.rank, factorization=factorization,
                                                        dtype=torch.cfloat, fixed_rank_modes=fixed_rank_modes,
                                                        **decomposition_kwargs)
            self.weight = self.weight.normal_(0, scale)
            self._get_weight = self._get_weight_factorized

        if bias:
            self.bias = nn.Parameter(scale * torch.randn(self.out_channels, 1, 1, 1))
        else:
            self.bias = 0

    def _get_weight_factorized(self, layer_index, corner_index):
        """Get the weights corresponding to a particular layer,
        corner of the Fourier coefficient (top=0 or bottom=1) -- corresponding to lower frequencies
        and complex_index (real=0 or imaginary=1)
        """
        return self.weight()[4*layer_index + corner_index, :, :, :, :, :].to_tensor().contiguous()

    def _get_weight_dense(self, layer_index, corner_index):
        """Get the weights corresponding to a particular layer,
        corner of the Fourier coefficient (top=0 or bottom=1) -- corresponding to lower frequencies
        and complex_index (real=0 or imaginary=1)
        """
        return self.weight[4*layer_index + corner_index, :, :, :, :, :]

    def forward(self, x, indices=0):
        with torch.autocast(device_type='cuda', enabled=False):
            batchsize, channels, height, width, depth = x.shape
            dtype = x.dtype
            # out_fft = torch.zeros(x.shape, device=x.device)

            #Compute Fourier coeffcients
            x = torch.fft.rfftn(x.float(), norm=self.fft_norm, dim=[-3, -2, -1])

            # Multiply relevant Fourier modes
            # x = torch.view_as_real(x)
            # The output will be of size (batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1)
            out_fft = torch.zeros([batchsize, self.out_channels,  height, width, depth//2 + 1], device=x.device, dtype=torch.cfloat)

            out_fft[:, :, :self.modes_height, :self.modes_width, :self.modes_depth] = contract_3D(
                x[:, :, :self.modes_height, :self.modes_width, :self.modes_depth], self._get_weight(indices, 0))
            out_fft[:, :, -self.modes_height:, :self.modes_width, :self.modes_depth] = contract_3D(
                x[:, :, -self.modes_height:, :self.modes_width, :self.modes_depth], self._get_weight(indices, 1))
            out_fft[:, :, self.modes_height:, -self.modes_width:, :self.modes_depth] = contract_3D(
                x[:, :, self.modes_height:, -self.modes_width:, :self.modes_depth], self._get_weight(indices, 2))
            out_fft[:, :, -self.modes_height:, -self.modes_width:, :self.modes_depth] = contract_3D(
                x[:, :, -self.modes_height:, -self.modes_width:, :self.modes_depth], self._get_weight(indices, 3))

            # out_size = (int(height*super_res), int(width*super_res))
            x = torch.fft.irfftn(out_fft, s=(height, width, depth), norm=self.fft_norm).type(dtype) #(x.size(-2), x.size(-1))) +
            x = x + self.bias

        if self.mlp is not None:
            x = self.mlp(x)

        return x

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution
        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError('A single convolution is parametrized, directly use the main class.')

        return SubConv(self, indices)

    def __getitem__(self, indices):
        return self.get_conv(indices)


class FactorizedSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_height, modes_width, n_layers=1, bias=True, scale='auto',
                 fft_norm='backward',
                 rank=0.5, factorization='cp', fixed_rank_modes=None, decomposition_kwargs=dict(), **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_height = modes_height
        self.modes_width = modes_width
        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers
        self.fft_norm = fft_norm

        if scale == 'auto':
            scale = (1 / (in_channels * out_channels))

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                fixed_rank_modes=[0]
            else:
                fixed_rank_modes=None

        if factorization is None:
            self.weight = nn.Parameter(scale * torch.randn(2*n_layers, in_channels, out_channels, self.modes_height, self.modes_width,
                                                            dtype=torch.cfloat))
            self._get_weight = self._get_weight_dense
        else:
            self.weight = tltorch.FactorizedTensor.new((2*n_layers, in_channels, out_channels, self.modes_height, self.modes_width),
                                                        rank=self.rank, factorization=factorization,
                                                        dtype=torch.cfloat, fixed_rank_modes=fixed_rank_modes,
                                                        **decomposition_kwargs)
            self.weight = self.weight.normal_(0, scale)
            self._get_weight = self._get_weight_factorized

        if bias:
            self.bias = nn.Parameter(scale * torch.randn(self.out_channels, 1, 1))
        else:
            self.bias = 0

    def _get_weight_factorized(self, layer_index, corner_index):
        """Get the weights corresponding to a particular layer,
        corner of the Fourier coefficient (top=0 or bottom=1) -- corresponding to lower frequencies
        and complex_index (real=0 or imaginary=1)
        """
        return self.weight()[2*layer_index + corner_index, :, :, :, : ].to_tensor().contiguous()

    def _get_weight_dense(self, layer_index, corner_index):
        """Get the weights corresponding to a particular layer,
        corner of the Fourier coefficient (top=0 or bottom=1) -- corresponding to lower frequencies
        and complex_index (real=0 or imaginary=1)
        """
        return self.weight[2*layer_index + corner_index, :, :, :, :]

    def forward(self, x, indices=0, super_res=1):
        with torch.autocast(device_type='cuda', enabled=False):
            batchsize, channels, height, width = x.shape
            dtype = x.dtype
            # out_fft = torch.zeros(x.shape, device=x.device)

            #Compute Fourier coeffcients
            x = torch.fft.rfft2(x.float(), norm=self.fft_norm)

            # Multiply relevant Fourier modes
            # x = torch.view_as_real(x)
            # The output will be of size (batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1)
            out_fft = torch.zeros([batchsize, self.out_channels,  height, width//2 + 1], device=x.device, dtype=torch.cfloat)

            # upper block (truncate high freq)
            out_fft[:, :, :self.modes_height, :self.modes_width:super_res] = contract_2D(x[:, :, :self.modes_height, :self.modes_width], self._get_weight(indices, 0))
            # Lower block
            out_fft[:, :, -self.modes_height:, :self.modes_width:super_res] = contract_2D(x[:, :, -self.modes_height:, :self.modes_width], self._get_weight(indices, 1))

            out_size = (int(height*super_res), int(width*super_res))
            x = torch.fft.irfft2(out_fft, s=out_size, norm=self.fft_norm).type(dtype) #(x.size(-2), x.size(-1)))

            return x + self.bias

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution
        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError('A single convolution is parametrized, directly use the main class.')

        return SubConv(self, indices)

    def __getitem__(self, indices):
        return self.get_conv(indices)


class SubConv(nn.Module):
    """Class representing one of the convolutions from the mother joint factorized convolution
    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to the same data,
    which is shared.
    """
    def __init__(self, main_conv, indices):
        super().__init__()
        self.main_conv = main_conv
        self.indices = indices

    def forward(self, x, **kwargs):
        return self.main_conv.forward(x, self.indices, **kwargs)


class FactorizedSpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, n_layers=1,
                 bias=True, scale='auto', fft_norm='forward', rank=0.5,
                 factorization='tucker', fixed_rank_modes=None, decomposition_kwargs=dict()):
        super().__init__()

        #Joint factorization only works for the same in and out channels
        if n_layers > 1:
            assert in_channels == out_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers
        self.fft_norm = fft_norm

        if scale == 'auto':
            scale = (1 / (in_channels * out_channels))

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                fixed_rank_modes=[0]
            else:
                fixed_rank_modes=None

        if factorization is None:
            self.weight = nn.Parameter(scale * torch.randn(n_layers, in_channels, out_channels, self.modes,
                                                            dtype=torch.cfloat))
            self._get_weight = self._get_weight_dense
        else:
            self.weight = tltorch.FactorizedTensor.new((n_layers, in_channels, out_channels, self.modes),
                                                        rank=self.rank, factorization=factorization,
                                                        dtype=torch.cfloat, fixed_rank_modes=fixed_rank_modes,
                                                        **decomposition_kwargs)
            self.weight.normal_(0, scale)
            self._get_weight = self._get_weight_factorized

        if bias:
            self.bias = nn.Parameter(scale * torch.randn(1, self.out_channels, 1))
        else:
            self.bias = 0

    def _get_weight_factorized(self, layer_index):
        #Get the weights corresponding to a particular layer
        return self.weight()[layer_index, :, :, : ].to_tensor().contiguous()

    def _get_weight_dense(self, layer_index):
        #Get the weights corresponding to a particular layer
        return self.weight[layer_index, :, :, :]

    def forward(self, x, indices=0, s=None):
        batchsize, channels, width = x.shape
        dtype = x.dtype

        if s is None:
            s = width

        #Compute Fourier coeffcients
        x = torch.fft.rfft(x, norm=self.fft_norm)

        # Multiply relevant Fourier modes
        out_fft = torch.zeros([batchsize, self.out_channels,  width//2 + 1], device=x.device, dtype=torch.cfloat)
        out_fft[:, :, :self.modes] = contract_1D(x[:, :, :self.modes], self._get_weight(indices))

        #Return to physical space
        x = torch.fft.irfft(out_fft, n=s, norm=self.fft_norm).type(dtype)

        return x + self.bias

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution
        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError('A single convolution is parametrized, directly use the main class.')

        return SubConv(self, indices)

    def __getitem__(self, indices):
        return self.get_conv(indices)



class JointFactorizedSpectralConv1d(nn.Module):
    def __init__(self, modes, width, n_layers=1, joint_factorization=True, in_channels=2, scale='auto',
                 non_linearity=nn.GELU, rank=1.0, factorization='tucker', bias=True,
                 fixed_rank_modes=False, fft_norm='forward', decomposition_kwargs=dict()):
        super().__init__()

        if isinstance(modes, int):
            self.modes = [modes for _ in range(n_layers)]
        else:
            self.modes = modes

        if isinstance(width, int):
            self.width = [width for _ in range(n_layers)]
        else:
            self.width = width

        assert len(self.width) == len(self.modes)

        self.n_layers = len(self.width)
        self.joint_factorization = joint_factorization

        if self.joint_factorization:
            assert self.width.count(self.width[0]) == self.n_layers and self.modes.count(self.modes[0]) == self.n_layers
            self.in_channels = self.width[0]
        else:
            self.in_channels = in_channels

        self.width = [self.in_channels] + self.width

        self.scale = scale
        self.non_linearity = non_linearity()
        self.rank = rank
        self.factorization = factorization
        self.bias = bias
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fft_norm = fft_norm

        if joint_factorization:
            self.convs = FactorizedSpectralConv1d(self.in_channels, self.width[0], self.modes[0],
                                                  n_layers=self.n_layers,
                                                  bias=self.bias,
                                                  scale=self.scale,
                                                  fft_norm=self.fft_norm,
                                                  rank=self.rank,
                                                  factorization=self.factorization,
                                                  fixed_rank_modes=self.fixed_rank_modes,
                                                  decomposition_kwargs=decomposition_kwargs)
        else:
            self.convs = nn.ModuleList([FactorizedSpectralConv1d(self.width[j], self.width[j+1], self.modes[j],
                                                                 n_layers=1,
                                                                 bias=self.bias,
                                                                 scale=self.scale,
                                                                 fft_norm=self.fft_norm,
                                                                 rank=self.rank,
                                                                 factorization=self.factorization,
                                                                 fixed_rank_modes=self.fixed_rank_modes,
                                                                 decomposition_kwargs=decomposition_kwargs) for j in range(self.n_layers)])

        self.linears = nn.ModuleList([nn.Conv1d(self.width[j], self.width[j+1], 1) for j in range(self.n_layers)])

    def forward(self, x, s=None):

        if s is None:
            s = [None for _ in range(self.n_layers)]

        if isinstance(s, int):
            s = [None for _ in range(self.n_layers-1)] + [s]

        for j in range(self.n_layers):
            x1 = self.convs[j](x, s=s[j])

            #Fourier interpolation
            if s[j] is not None:
                x2 = torch.fft.irfft(torch.fft.rfft(x, norm=self.fft_norm), n=s[j], norm=self.fft_norm)
            else:
                x2 = x

            x2 = self.linears[j](x2)

            x = x1 + x2

            if j < (self.n_layers - 1):
                x = self.non_linearity(x)

        return x




class FactorizedFNO3d(nn.Module):
    def __init__(self, modes_height, modes_width,  modes_depth, width, fc_channels=256, n_layers=4,
                joint_factorization=True, non_linearity=F.gelu,
                rank=1.0, factorization='cp', fixed_rank_modes=False,
                domain_padding=9, in_channels=3, Block=None,
                verbose=True, fft_contraction='complex',
                fft_norm='backward',
                mlp=False,
                decomposition_kwargs=dict()):
        super().__init__()
        self.modes_height = modes_height
        self.modes_width = modes_width
        self.modes_depth = modes_depth
        self.width = width
        self.fc_channels = fc_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.domain_padding = domain_padding # pad the domain if input is non-periodic
        self.in_channels = in_channels
        self.decomposition_kwargs = decomposition_kwargs
        self.fft_norm = fft_norm
        self.verbose = verbose

        if Block is None:
            Block = FactorizedSpectralConv3d
        if verbose:
            print(f'FNO Block using {Block}, fft_contraction={fft_contraction}')

        self.Block = Block

        if joint_factorization:
            self.convs = Block(self.width, self.width, self.modes_height, self.modes_width, self.modes_depth,
                               rank=rank,
                               fft_contraction=fft_contraction,
                               fft_norm=fft_norm,
                               factorization=factorization,
                               fixed_rank_modes=fixed_rank_modes,
                               decomposition_kwargs=decomposition_kwargs,
                               mlp=mlp,
                               n_layers=n_layers)
        else:
            self.convs = nn.ModuleList([Block(self.width, self.modes_height, self.modes_width, self.modes_depth,
                                              fft_contraction=fft_contraction,
                                              rank=rank,
                                              factorization=factorization,
                                              fixed_rank_modes=fixed_rank_modes,
                                              decomposition_kwargs=decomposition_kwargs,
                                              mlp=mlp,
                                              n_layers=1) for _ in range(n_layers)])
        self.linears = nn.ModuleList([nn.Conv3d(self.width, self.width, 1) for _ in range(n_layers)])

        self.fc0 = nn.Linear(in_channels, self.width) # input channel is 3: (a(x, y), x, y)
        self.fc1 = nn.Linear(self.width, fc_channels)
        self.fc2 = nn.Linear(fc_channels, 1)

    def forward(self, x, super_res=1):
        #grid = self.get_grid(x.shape, x.device)
        #x = torch.cat((x, grid), dim=-1)
        #x = self.fc0(x)
        #x = x.permute(0, 3, 1, 2)

        x = x.permute(0,2,3,4,1)
        x = self.fc0(x)
        x = x.permute(0,4,1,2,3)

        x = F.pad(x, [0, self.domain_padding])

        for i in range(self.n_layers):
            if super_res > 1 and i == (self.n_layers - 1):
                super_res = super_res
            else:
                super_res = 1

            x1 = self.convs[i](x) #, super_res=super_res)
            x2 = self.linears[i](x)
            x = x1 + x2
            if i < (self.n_layers - 1):
                x = self.non_linearity(x)

        x = x[..., :-self.domain_padding]
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        x = x.permute(0,4,1,2,3)
        return x


class FactorizedFNO2d(nn.Module):
    def __init__(self, modes_height, modes_width,  width, fc_channels=256, n_layers=4,
                joint_factorization=True, non_linearity=F.gelu,
                rank=1.0, factorization='cp', fixed_rank_modes=False,
                domain_padding=9, in_channels=3, Block=None,
                verbose=True, fft_contraction='complex',
                fft_norm='backward',
                decomposition_kwargs=dict()):
        super().__init__()
        """
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes_height = modes_height
        self.modes_width = modes_width
        self.width = width
        self.fc_channels = fc_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.domain_padding = domain_padding # pad the domain if input is non-periodic
        self.in_channels = in_channels
        self.decomposition_kwargs = decomposition_kwargs
        self.fft_norm = fft_norm
        self.verbose = verbose

        if Block is None:
            Block = FactorizedSpectralConv2d
        if verbose:
            print(f'FNO Block using {Block}, fft_contraction={fft_contraction}')

        self.Block = Block

        if joint_factorization:
            self.convs = Block(self.width, self.width, self.modes_height, self.modes_width,
                               rank=rank,
                               fft_contraction=fft_contraction,
                               fft_norm=fft_norm,
                               factorization=factorization,
                               fixed_rank_modes=fixed_rank_modes,
                               decomposition_kwargs=decomposition_kwargs,
                               n_layers=n_layers)
        else:
            self.convs = nn.ModuleList([Block(self.width, self.width, self.modes_height,
                                              self.modes_width,
                                              fft_contraction=fft_contraction,
                                              rank=rank,
                                              factorization=factorization,
                                              fixed_rank_modes=fixed_rank_modes,
                                              decomposition_kwargs=decomposition_kwargs,
                                              n_layers=1) for _ in range(n_layers)])
        self.linears = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(n_layers)])

        self.fc0 = nn.Linear(in_channels, self.width) # input channel is 3: (a(x, y), x, y)
        self.fc1 = nn.Linear(self.width, fc_channels)
        self.fc2 = nn.Linear(fc_channels, 1)

    def forward(self, x, super_res=1):
        #grid = self.get_grid(x.shape, x.device)
        #x = torch.cat((x, grid), dim=-1)
        #x = self.fc0(x)
        #x = x.permute(0, 3, 1, 2)

        x = x.permute(0,2,3,1)
        x = self.fc0(x)
        x = x.permute(0,3,1,2)

        x = F.pad(x, [0, self.domain_padding, 0, self.domain_padding])

        for i in range(self.n_layers):
            if super_res > 1 and i == (self.n_layers - 1):
                super_res = super_res
            else:
                super_res = 1

            x1 = self.convs[i](x) #, super_res=super_res)
            x2 = self.linears[i](x)
            x = x1 + x2
            if i < (self.n_layers - 1):
                x = self.non_linearity(x)

        x = x[..., :-self.domain_padding, :-self.domain_padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        x = x.permute(0,3,1,2)
        return x

    # def extra_repr(self):
    #     s = (f'{self.modes_height=}, {self.modes_width=},  {self.width=}, {self.fc_channels=}, {self.n_layers=}, '
    #          f'{self.joint_factorization=}, {self.non_linearity=}, '
    #          f'{self.rank=}, {self.factorization=}, {self.fixed_rank_modes=}, '
    #          f'{self.domain_padding=}, {self.in_channels=}, {self.Block=}, '
    #          f'{self.verbose=}, '
    #          f'{self.decomposition_kwargs=}')
    #     return s


class FactorizedFNO1d(nn.Module):
    def __init__(self, modes, width, in_channels=2, out_channels=1, n_layers=4,
                 lifting=None, projection=None, joint_factorization=True,  scale='auto',
                 non_linearity=nn.GELU, rank=1.0, factorization='tucker', bias=True,
                 fixed_rank_modes=False, fft_norm='forward', decomposition_kwargs=dict()):
        super().__init__()

        if isinstance(width, int):
            init_width = width
            final_width = width
        else:
            init_width = width[0]
            final_width = width[-1]

        self.non_linearity = non_linearity()

        if lifting is None:
            self.lifting = nn.Linear(in_channels, init_width)

        if projection is None:
            self.projection = nn.Sequential(nn.Linear(final_width, 256),
                                            self.non_linearity,
                                            nn.Linear(256, out_channels))

        self.fno_layers = JointFactorizedSpectralConv1d(modes, width, n_layers=n_layers, joint_factorization=joint_factorization,
                                                        in_channels=init_width, scale=scale, non_linearity=non_linearity,
                                                        rank=rank, factorization=factorization, bias=bias, fixed_rank_modes=fixed_rank_modes,
                                                        fft_norm=fft_norm, decomposition_kwargs=decomposition_kwargs)

    def forward(self, x, s=None):
        #Lifting
        x = x.permute(0,2,1)
        x = self.lifting(x)
        x = x.permute(0,2,1)

        #Fourier layers
        x = self.fno_layers(x, s=s)

        #Projection
        x = x.permute(0,2,1)
        x = self.projection(x)
        x = x.permute(0,2,1)

        return x

