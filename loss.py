import torch

from torch import Tensor
import torch.nn.functional as F

from pycox.models import utils, loss
from pycox.evaluation import EvalSurv


def nll_pc_hazard_loss(phi: Tensor, idx_durations: Tensor, events: Tensor, interval_frac: Tensor,
                       reduction: str = 'mean') -> Tensor:
    """Negative log-likelihood of the PC-Hazard parametrization model [1].

    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where hazard = sigmoid(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same length as 'idx_durations'.
        interval_frac {torch.tensor} -- Fraction of last interval before event/censoring.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.

    Returns:
        torch.tensor -- The negative log-likelihood.
    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    if events.dtype is torch.bool:
        events = events.float()
    idx_durations = idx_durations.view(-1, 1)
    events = events.view(-1)
    interval_frac = interval_frac.view(-1)

    keep = idx_durations.view(-1) >= 0
    phi = phi[keep, :]
    idx_durations = idx_durations[keep, :]
    events = events[keep]
    interval_frac = interval_frac[keep]

    # log_h_e = F.softplus(phi.gather(1, idx_durations).view(-1)).log().mul(events)
    log_h_e = utils.log_softplus(phi.gather(1, idx_durations).view(-1)).mul(events)
    haz = F.softplus(phi)
    scaled_h_e = haz.gather(1, idx_durations).view(-1).mul(interval_frac)
    haz = utils.pad_col(haz, where='start')
    sum_haz = haz.cumsum(1).gather(1, idx_durations).view(-1)
    loss = - log_h_e.sub(scaled_h_e).sub(sum_haz)
    return _reduction(loss, reduction)


class BrierLoss(loss._Loss):
    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor, interval_frac: Tensor, reduction: str = 'mean') -> Tensor:
        """Negative log-likelihood of the PC-Hazard parametrization model.
        See `loss.nll_pc_hazard_loss` for details.

        Arguments:
            reduction {string} -- How to reduce the loss.
                'none': No reduction.
                'mean': Mean of tensor.
                'sum: sum.

        Returns:
            torch.tensor -- The negative log-likelihood loss.
        """
        x = 1
        # ev = EvalSurv(phi.detach().numpy(), idx_durations.numpy(), events.numpy(), censor_surv='km')
        x = x + 1
        ## return loss._reduction(ev.brier_score(time_grid), reduction)
        return nll_pc_hazard_loss(phi, idx_durations, events, interval_frac, self.reduction)
