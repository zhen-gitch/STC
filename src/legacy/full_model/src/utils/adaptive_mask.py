import torch
import torch.nn as nn


class FineGrainedChannelAdaptiveMask(nn.Module):
    """
    根据辅助回归损失对时序特征的梯度，生成通道级或时间级对抗掩码。
    """

    def __init__(self, base_mask_prob=0.05, max_mask_prob=0.3, share_feature_mask=True, use_temporal_mask=False):
        super().__init__()
        self.base_mask_prob = base_mask_prob
        self.max_mask_prob = max_mask_prob
        self.share_feature_mask = share_feature_mask
        self.use_temporal_mask = use_temporal_mask

    def compute_masks(self, features, auxiliary_loss_fn, model, mask):
        if not features.requires_grad:
            raise RuntimeError("FineGrainedChannelAdaptiveMask: features 必须 requires_grad=True。")

        with torch.enable_grad():
            time_seq_feature = model.temporal_encoder(features, mask)
            time_feature_pool = model.encode_and_pool(time_seq_feature, mask)
            reg_pool, _, _ = model.cgc_layer(time_feature_pool)
            bdi_pred = model.reg_task_head(reg_pool).squeeze(-1)
            loss_aux = auxiliary_loss_fn(bdi_pred)

            grad = torch.autograd.grad(
                outputs=loss_aux,
                inputs=features,
                retain_graph=False,
                create_graph=False,
                only_inputs=True
            )[0].detach()

        if self.use_temporal_mask:
            temporal_imp = torch.norm(grad, dim=-1)
            t_max = temporal_imp.max(dim=1, keepdim=True)[0]
            t_min = temporal_imp.min(dim=1, keepdim=True)[0]
            t_norm = (temporal_imp - t_min) / (t_max - t_min + 1e-8)

            adaptive_prob_t = self.base_mask_prob + t_norm * (
                self.max_mask_prob - self.base_mask_prob
            )
            adaptive_prob_t = adaptive_prob_t.unsqueeze(-1)
            temporal_mask = (torch.rand_like(adaptive_prob_t) > adaptive_prob_t).float().detach()
        else:
            temporal_mask = torch.ones(
                features.shape[0],
                features.shape[1],
                1,
                device=features.device,
                dtype=features.dtype
            )
            adaptive_prob_t = torch.zeros_like(temporal_mask)

        if self.share_feature_mask:
            feature_imp = torch.norm(grad, dim=(0, 1))
            f_max = feature_imp.max()
            f_min = feature_imp.min()
            f_norm = (feature_imp - f_min) / (f_max - f_min + 1e-8)
            adaptive_prob_f = self.base_mask_prob + f_norm * (
                self.max_mask_prob - self.base_mask_prob
            )
            adaptive_prob_f = adaptive_prob_f.view(1, 1, -1)
        else:
            feature_imp = torch.norm(grad, dim=1)
            f_max = feature_imp.max(dim=1, keepdim=True)[0]
            f_min = feature_imp.min(dim=1, keepdim=True)[0]
            f_norm = (feature_imp - f_min) / (f_max - f_min + 1e-8)
            adaptive_prob_f = self.base_mask_prob + f_norm * (
                self.max_mask_prob - self.base_mask_prob
            )
            adaptive_prob_f = adaptive_prob_f.unsqueeze(1)

        feature_mask = (torch.rand_like(adaptive_prob_f) > adaptive_prob_f).float().detach()
        return temporal_mask, feature_mask, adaptive_prob_t, adaptive_prob_f
