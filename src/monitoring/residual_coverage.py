import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_residual_coverage(
    gt_frame: torch.Tensor,
    pred_frame: torch.Tensor,
    prev_frame: torch.Tensor,
    changed_threshold: float = 1e-3,
) -> dict[str, float]:
    """
    Measure how much of the frame-to-frame residual the model captures.

    All inputs are [B, C, H, W].  Values in [-1, 1] are normalised to [0, 1]
    before computation so thresholds / MSE values are comparable across runs.

    Returns a dict with:
        residual_r2             1 - MSE(pred, gt) / MSE(prev, gt)
        residual_cosine         mean cosine similarity of residual vectors
        changed_pixel_mse       MSE restricted to pixels where |gt - prev| > threshold
        pred_mse                MSE(pred, gt) over all pixels
        copy_prev_mse           MSE(prev, gt) over all pixels
        changed_pixel_fraction  fraction of pixels that changed
    """
    gt = gt_frame.float()
    pred = pred_frame.float()
    prev = prev_frame.float()

    if gt.min() < 0:
        gt = (gt + 1) / 2
    if pred.min() < 0:
        pred = (pred + 1) / 2
    if prev.min() < 0:
        prev = (prev + 1) / 2

    gt = gt.clamp(0, 1)
    pred = pred.clamp(0, 1)
    prev = prev.clamp(0, 1)

    pred_mse = F.mse_loss(pred, gt).item()
    copy_prev_mse = F.mse_loss(prev, gt).item()

    if copy_prev_mse < 1e-10:
        residual_r2 = 1.0 if pred_mse < 1e-10 else 0.0
    else:
        residual_r2 = 1.0 - pred_mse / copy_prev_mse

    r_gt = (gt - prev).flatten(1)
    r_pred = (pred - prev).flatten(1)
    cosine = F.cosine_similarity(r_pred, r_gt, dim=1).mean().item()

    changed_mask = (gt - prev).abs() > changed_threshold  # [B, C, H, W]
    changed_pixel_fraction = changed_mask.float().mean().item()

    if changed_mask.any():
        changed_pixel_mse = F.mse_loss(pred[changed_mask], gt[changed_mask]).item()
    else:
        changed_pixel_mse = 0.0

    return {
        "residual_r2": residual_r2,
        "residual_cosine": cosine,
        "changed_pixel_mse": changed_pixel_mse,
        "pred_mse": pred_mse,
        "copy_prev_mse": copy_prev_mse,
        "changed_pixel_fraction": changed_pixel_fraction,
    }
