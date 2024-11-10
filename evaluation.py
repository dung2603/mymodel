# evaluation.py

import numpy as np
import torch
import torch.nn as nn

def compute_errors(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'."""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = np.sqrt(np.mean((gt - pred) ** 2))

    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = np.mean(np.abs(np.log10(gt) - np.log10(pred)))

    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel,
                rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

def compute_metrics(gt, pred, min_depth=0.1, max_depth=10.0):
    """Compute metrics of predicted depth maps."""
    if gt.shape != pred.shape:
        pred = nn.functional.interpolate(
            pred.unsqueeze(0).unsqueeze(0),
            size=gt.shape,
            mode='bilinear',
            align_corners=True).squeeze()

    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()

    pred[pred < min_depth] = min_depth
    pred[pred > max_depth] = max_depth

    mask = np.logical_and(gt > min_depth, gt < max_depth)

    pred = pred[mask]
    gt = gt[mask]

    return compute_errors(gt, pred)
