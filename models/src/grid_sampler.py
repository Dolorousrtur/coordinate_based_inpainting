import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def profile(f):
    return f


@profile
def meshgrid_tensor(*sizes, normalize=True, device='cuda:0'):
    # returns tensor of size (*sizes, len(sizes))
    if not normalize:
        aranges = [torch.arange(cur_size, device=device) for cur_size in sizes]
        grids = torch.meshgrid(aranges)
        grid = torch.stack(grids, dim=-1)
        #grid = grid.float()    # this operation might be excessive
    else:
        aranges = [torch.arange(cur_size, device=device).float() for cur_size in sizes]
        grids = torch.meshgrid(aranges)
        grid = np.stack([cur_grid / float(max(sizes[i] - 1, 1)) for i, cur_grid in enumerate(grids)], 
                        dim=-1)
        #grid = torch.stack([cur_grid / float(sizes[i]) for i, cur_grid in enumerate(grids)], 
        #                  dim=-1)
        #grid = grid.float()
    return grid

@profile
def ravel_multi_index(indices, shape):
    # Gorner scheme:
    indices_ravel = indices[0]
    for i in range(1, len(indices)):
        indices_ravel = indices_ravel * shape[i] + indices[i]
    return indices_ravel

@profile
def add_repeated(t, indices, values):
    """Performs an in-place operation t[indices[0], indices[1], ..., indices[-1]] += values 
    such that for each repeated cell in multi-index `indices` all values will be accounted 
    in the summation. E.g. see:
    
    >>> A = torch.zeros(5)
    >>> A[[1, 1, 2]] += 1
    >>> A
    tensor([ 0.,  1.,  1.,  0.,  0.])
    
    >>> A = torch.zeros(5)
    >>> add_repeated(A, ([1, 1, 2],), 1)
    >>> A
    tensor([ 0.,  2.,  1.,  0.,  0.])
    
    PyTorch has a function torch.Tensor.index_add_() which solves this task only for flattened arrays. 
    This is an adaptation for multi-dimensional arrays.
    
    Accepts:
    * t: torch Tensor of shape (N_1, N_2, ..., N_K)
    * indices: tuple of LongTensors, all of shape (M,); indices[i] must have values in {0, 1, ..., N_i - 1}
    * values: torch Tensor of shape (M,)
    Returns: None."""
    
    n_dims = len(indices)
    shape = t.shape
    t_ravel = t.view(t.numel())    # flatten
    indices_ravel = ravel_multi_index(indices, shape)
    t_ravel.index_add_(0, indices_ravel, values)
    t = t_ravel.view(*shape)


class InvGridSamplerNumerator(nn.Module):
    eps = 1e-10
    
    def __init__(self, OH=None, OW=None):
        super(InvGridSamplerNumerator, self).__init__()
        self.OH = OH
        self.OW = OW
    
    @profile
    def forward(self, x, inv_grid):
        eps = InvGridSamplerNumerator.eps
        batch_size, n_channels, h, w = x.size(0), x.size(1), \
            x.size(2) if self.OH is None else self.OH, \
            x.size(3) if self.OW is None else self.OW
        inv_grid = (inv_grid.clone() + 1) / 2.0
        # inv_grid[..., 0] *= max(h - 1, 1)
        # inv_grid[..., 1] *= max(w - 1, 1)
        inv_grid[..., 0] *= h
        inv_grid[..., 1] *= w

        inv_grid += 1    # we convert [0, h] and [0, w] coordinate ranges to [1, h + 1], [1, w + 1]
        inv_grid = torch.stack([inv_grid[..., 0].clamp(0, h + 1 - 2 * eps),
                                inv_grid[..., 1].clamp(0, w + 1 - 2 * eps)], dim=-1)

        inv_grid = inv_grid[:, np.newaxis].repeat(1, n_channels, 1, 1, 1)
        inv_grid_long = (inv_grid + eps).long()
        #A = torch.zeros((batch_size, n_channels, h, w), device=x.device)
        A = torch.zeros((batch_size, n_channels, h + 3, w + 3), device=x.device)
        mgrid = meshgrid_tensor(batch_size, n_channels, x.size(2), x.size(3), normalize=False, device=inv_grid.device)

        input_cells = mgrid.view(-1, mgrid.size(4))
        input_inds_b, input_inds_ch, input_inds_i, input_inds_j = \
            input_cells[..., 0], input_cells[..., 1], input_cells[..., 2], input_cells[..., 3]

        output_inds_b  = input_inds_b
        output_inds_ch = input_inds_ch
        #output_inds_i  = (inv_grid_long[..., 0] + di)[input_mask_for_grid]
        #output_inds_j  = (inv_grid_long[..., 1] + dj)[input_mask_for_grid]
        #output_cells_float = inv_grid[input_inds_b, input_inds_ch, input_inds_i, input_inds_j]
        output_cells_float = inv_grid.view(-1, inv_grid.size(4))
        #output_cells_long = inv_grid_long[input_inds_b, input_inds_ch, input_inds_i, input_inds_j]
        output_cells_long = output_cells_float.long()

        for di in range(0, 2):
            output_inds_i = output_cells_long[..., 0] + di
            corners_i = output_inds_i.float()
            bilinear_weights_i = F.relu(1 - torch.abs((output_cells_float[..., 0] - corners_i)))
            for dj in range(0, 2):
                output_inds_j = output_cells_long[..., 1] + dj
                
                # corners_i = inv_grid_long[input_inds_b, input_inds_ch, input_inds_i, input_inds_j, 0].float() + di
                # corners_j = inv_grid_long[input_inds_b, input_inds_ch, input_inds_i, input_inds_j, 1].float() + dj
                
                corners_j = output_inds_j.float()
                # bilinear_weights = \
                #     F.relu(1 - torch.abs((inv_grid[input_inds_b, input_inds_ch, input_inds_i, input_inds_j, 0] - corners_i))) * \
                #     F.relu(1 - torch.abs((inv_grid[input_inds_b, input_inds_ch, input_inds_i, input_inds_j, 1] - corners_j)))
                
                bilinear_weights = \
                    bilinear_weights_i * \
                    F.relu(1 - torch.abs((output_cells_float[..., 1] - corners_j)))

                add_repeated(A, (output_inds_b, output_inds_ch, output_inds_i, output_inds_j), 
                    x.view(-1) * bilinear_weights)
        A = A[..., 1:h + 1, 1:w + 1]    # cutting out the border
        return A
    

class InvGridSamplerDenominator(nn.Module):
    eps = 1e-10
    
    def __init__(self, OH=None, OW=None):
        super(InvGridSamplerDenominator, self).__init__()
        self.OH = OH
        self.OW = OW
    
    @profile
    def forward(self, x, inv_grid):
        eps = InvGridSamplerDenominator.eps
        batch_size, n_channels, h, w = x.size(0), x.size(1), \
            x.size(2) if self.OH is None else self.OH, \
            x.size(3) if self.OW is None else self.OW
        inv_grid = (inv_grid.clone() + 1) / 2.0
        # inv_grid[..., 0] *= max(h - 1, 1)
        # inv_grid[..., 1] *= max(w - 1, 1)
        inv_grid[..., 0] *= h
        inv_grid[..., 1] *= w

        inv_grid += 1    # we convert [0, h] and [0, w] coordinate ranges to [1, h + 1], [1, w + 1]
        inv_grid = torch.stack([inv_grid[..., 0].clamp(0, h + 1 - 2 * eps),
                                inv_grid[..., 1].clamp(0, w + 1 - 2 * eps)], dim=-1)

        inv_grid_long = (inv_grid + eps).long()
        B = torch.zeros((batch_size, n_channels, h + 3, w + 3), device=x.device)
        # mgrid = meshgrid_tensor(batch_size, x.size(2), x.size(3), normalize=False).to(inv_grid.device)
        mgrid = meshgrid_tensor(batch_size, x.size(2), x.size(3), normalize=False, device=inv_grid.device)

        #input_mask_for_grid = (0 <= inv_grid[..., 0] + di) & (inv_grid[..., 0] + di < h) & \
        #input_mask_for_grid = input_mask_for_grid_half & \
        #     (0 <= inv_grid[..., 1] + dj) & (inv_grid[..., 1] + dj < w)
        
        # input_inds_b   = mgrid[..., 0][input_mask_for_grid]
        # input_inds_i   = mgrid[..., 1][input_mask_for_grid]
        # input_inds_j   = mgrid[..., 2][input_mask_for_grid]
        #input_cells = mgrid[input_mask_for_grid]
        input_cells = mgrid.view(-1, mgrid.size(3))
        input_inds_b, input_inds_i, input_inds_j = input_cells[:, 0], input_cells[:, 1], input_cells[:, 2]
        output_inds_b  = input_inds_b
        # output_inds_i  = (inv_grid_long[..., 0] + di)[input_mask_for_grid]
        # output_inds_j  = (inv_grid_long[..., 1] + dj)[input_mask_for_grid]
        # output_cells_float = inv_grid[input_inds_b, input_inds_i, input_inds_j]
        output_cells_float = inv_grid.view(-1, inv_grid.size(3))
        output_cells_long = output_cells_float.long()

        for di in range(0, 2):
            output_inds_i = output_cells_long[..., 0] + di
            corners_i = output_inds_i.float()
            bilinear_weights_i = F.relu(1 - torch.abs((output_cells_float[..., 0] - corners_i)))
            for dj in range(0, 2):
                output_inds_j = output_cells_long[..., 1] + dj
                
                # corners_i = inv_grid_long[input_inds_b, input_inds_i, input_inds_j, 0].float() + di
                # corners_j = inv_grid_long[input_inds_b, input_inds_i, input_inds_j, 1].float() + dj
                
                corners_j = output_inds_j.float()
                # bilinear_weights = \
                #     F.relu(1 - torch.abs((inv_grid[input_inds_b, input_inds_i, input_inds_j, 0] - corners_i))) * \
                #     F.relu(1 - torch.abs((inv_grid[input_inds_b, input_inds_i, input_inds_j, 1] - corners_j)))
                
                bilinear_weights = \
                    bilinear_weights_i * \
                    F.relu(1 - torch.abs((output_cells_float[..., 1] - corners_j)))

                B_ch = torch.zeros_like(B[:, 0])
                add_repeated(B_ch, (output_inds_b, output_inds_i, output_inds_j), bilinear_weights)
                B += B_ch[:, np.newaxis]
        B = B[..., 1:h + 1, 1:w + 1]
        return B

    
class InvGridSamplerDecomposed(nn.Module):
    eps = 1e-10
    
    def __init__(self, OH=None, OW=None, return_A=False, return_B=False,
                 hole_fill_color=1):
        super(InvGridSamplerDecomposed, self).__init__()
        self.OH = OH
        self.OW = OW
        self.numerator = InvGridSamplerNumerator(OH=OH, OW=OW)
        self.denominator = InvGridSamplerDenominator(OH=OH, OW=OW)
        self.return_A = return_A
        self.return_B = return_B
        self.hole_fill_color = hole_fill_color
    
    @profile
    def forward(self, x, inv_grid):
        # x          -- tensor of size (batch_size, in_channels, IH, IW)
        # inv_grid   -- tensor of size (batch_size, IH, IW, 2) with values strictly in [-1, 1]
        
        eps = InvGridSamplerDecomposed.eps
        A = self.numerator(x, inv_grid)
        B = self.denominator(x, inv_grid)
        sampled = (A / (B + eps)) * (B > eps).float() + self.hole_fill_color * (B <= eps).float()
        if self.return_A and self.return_B:
            return sampled, A, B
        if self.return_A:
            return sampled, A
        if self.return_B:
            return sampled, B
        return sampled

    
# if __name__ == '__main__':
#     import time

#     sampler = InvGridSamplerDecomposed()

#     batch_size, n_channels, height, width = 10, 3, 128, 128
#     torch.manual_seed(999)
#     temp_x = torch.rand(batch_size, n_channels, height, width).requires_grad_().cuda()
    
#     temp_grid = meshgrid_tensor(height, width).requires_grad_().cuda()
#     temp_grid = (temp_grid + 0.01) * 2
#     temp_grid = temp_grid ** 2
#     temp_grid = (temp_grid - 0.5) * 2
#     temp_grid = temp_grid[np.newaxis]
#     #print('x:')
#     #print(temp_x)
#     temp_grid = temp_grid.repeat(batch_size, 1, 1, 1)

#     print('layer output:')
#     start_time = time.time()
#     print(sampler(temp_x, temp_grid))
#     end_time = time.time()
#     print('Forward pass time: {:.3} s'.format(end_time - start_time))
#     #TODO check backward pass time

#     # Testing time based on several observations
#     time_tries = []
#     n_tries = 5
#     for try_no in range(n_tries):
#         temp_x = torch.rand(batch_size, n_channels, height, width).requires_grad_().cuda()
#         temp_grid = meshgrid_tensor(height, width).requires_grad_().cuda()
#         temp_grid = (temp_grid + 0.01) * 2
#         temp_grid = temp_grid ** (1 + torch.rand(1).cuda())
#         temp_grid = (temp_grid - 0.5) * 2
#         temp_grid = temp_grid[np.newaxis]
#         #print('x:')
#         #print(temp_x)
#         temp_grid = temp_grid.repeat(batch_size, 1, 1, 1)

#         start_time = time.time()
#         sampler(temp_x, temp_grid)
#         end_time = time.time()
#         time_tries.append(end_time - start_time)
#     print('Forward pass times based on several repeats:', time_tries)
#     print('Forward pass time, averaged: {:.3} s'.format(np.mean(time_tries)))
    
#     """
#     print('grad check:')

#     from torch.autograd import gradcheck

#     test = gradcheck(sampler.apply, (temp_x, temp_grid), eps=1e-3)
#     print(test)
#     """