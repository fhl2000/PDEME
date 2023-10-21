import torch

def MSE(pred, target):
    """return mean square error or root mean square error in a batch

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape)
    target = target.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
    nb, nc, nt = pred.shape[0], pred.shape[1], pred.shape[-1]
    errors = pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt]) # 在空间维度上取平均
    res = torch.mean(errors**2, dim=2) # (bs, v, t)
    return torch.mean(res, dim=0) # (v, t) batch中取平均


def RMSE(pred, target):
    return torch.sqrt(MSE(pred, target))


def bMSE(pred, target):
    """return mean square error or root mean square error in a batch

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape)
    target = target.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
    nb, nc, nt = pred.shape[0], pred.shape[1], pred.shape[-1]
    spatial_dim = len(pred.shape) - 3
    if spatial_dim == 1:
        bd_square_error = (pred[:, :, 0, :] - target[:, :, 0, :])**2
        bd_square_error += (pred[:, :, -1, :] - target[:, :, -1, :])**2 # (bs, v, t)
        bd_mean_square_error = bd_square_error / 2.
    elif spatial_dim == 2:
        bd_x_square_error = (pred[:, :, 0, :, :] - target[:, :, 0, :, :])**2
        bd_x_square_error += (pred[:, :, -1, :, :] - target[:, :, -1, :, :])**2
        bd_x_mean_square_error = torch.mean(bd_x_square_error / 2., dim=2) # (bs, v, t)
        bd_y_square_error = (pred[:, :, :, 0, :] - target[:, :, :, 0, :])**2
        bd_y_square_error += (pred[:, :, :, -1, :] - target[:, :, :, -1, :])**2
        bd_y_mean_square_error = torch.mean(bd_y_square_error / 2., dim=2) # (bs, v, t)
        bd_mean_square_error = (bd_x_mean_square_error + bd_y_mean_square_error) / 2.
    else: # spatial_dim == 3
        bd_x_square_error = (pred[:, :, 0, :, :, :] - target[:, :, 0, :, :, :])**2
        bd_x_square_error += (pred[:, :, -1, :, :, :] - target[:, :, -1, :, :, :])**2
        bd_x_mean_square_error = torch.mean(bd_x_square_error.view([nb, nc, -1, nt]) / 2., dim=2) # (bs, v, t)
        bd_y_square_error = (pred[:, :, :, 0, :, :] - target[:, :, :, 0, :, :])**2
        bd_y_square_error += (pred[:, :, :, -1, :, :] - target[:, :, :, -1, :, :])**2
        bd_y_mean_square_error = torch.mean(bd_y_square_error.view([nb, nc, -1, nt]) / 2., dim=2) # (bs, v, t)
        bd_z_square_error = (pred[:, :, :, :, 0, :] - target[:, :, :, :, 0, :])**2
        bd_z_square_error += (pred[:, :, :, :, -1, :] - target[:, :, :, :, -1, :])**2
        bd_z_mean_square_error = torch.mean(bd_y_square_error.view([nb, nc, -1, nt]) / 2., dim=2) # (bs, v, t)
        bd_mean_square_error = (bd_x_mean_square_error + bd_y_mean_square_error + bd_z_mean_square_error) / 3.
    return torch.mean(bd_mean_square_error, dim=0) # (v, t)


def bRMSE(pred, target):
    return torch.sqrt(bMSE(pred, target))


def L2RE(pred, target):
    """l2 relative error (nRMSE in PDEBench)

    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    squared: bool, default=True. If True returns MSE value, if False returns RMSE value.
    """
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape)
    target = target.permute(temp_shape)
    nb, nc, nt = pred.shape[0], pred.shape[1], pred.shape[-1]
    errors = pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt])
    res = torch.sum(errors**2, dim=2) / torch.sum(target.view([nb, nc, -1, nt])**2, dim=2)
    return torch.mean(torch.sqrt(res), dim=0) # (v, t)


def L1RE(pred, target):
    """l1 relative error (backup)

    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    squared: bool, default=True. If True returns MSE value, if False returns RMSE value.
    """
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape)
    target = target.permute(temp_shape)
    nb, nc, nt = pred.shape[0], pred.shape[1], pred.shape[-1]
    l1_errors = torch.abs(pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt]))
    res = torch.sum(l1_errors, dim=2) / torch.sum(torch.abs(target.view([nb, nc, -1, nt])), dim=2)
    return torch.mean(res, dim=0)


def MaxError(pred, target):
    """return max error in a batch

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    errors = torch.abs(pred - target)
    nc = errors.shape[-1]
    res, _ = torch.max(errors.view([-1, nc]), dim=0) # retain the last dim
    return res