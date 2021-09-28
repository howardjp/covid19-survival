## Stolen from https://github.com/xuyxu/Soft-Decision-Tree

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox import models
from pycox.models.utils import pad_col, make_subgrid
from pycox.preprocessing import label_transforms
from pycox.models.interpolation import InterpolateLogisticHazard

class SDTHazard(models.base.SurvBase):
    """
    A discrete-time survival model that minimize the likelihood for right-censored data by
    parameterizing the hazard function. Also known as  "Nnet-survival" [3].

    The Logistic-Hazard was first proposed by [2], but this implementation follows [1].

    Arguments:
        net {torch.nn.Module} -- A torch module.

    Keyword Arguments:
        optimizer {Optimizer} -- A torch optimizer or similar. Preferably use torchtuples.optim instead of
            torch.optim, as this allows for reinitialization, etc. If 'None' set to torchtuples.optim.AdamW.
            (default: {None})
        device {str, int, torch.device} -- Device to compute on. (default: {None})
            Preferably pass a torch.device object.
            If 'None': use default gpu if available, else use cpu.
            If 'int': used that gpu: torch.device('cuda:<device>').
            If 'string': string is passed to torch.device('string').
        duration_index {list, np.array} -- Array of durations that defines the discrete times.
            This is used to set the index of the DataFrame in `predict_surv_df`.

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf

    [2] Charles C. Brown. On the use of indicator variables for studying the time-dependence of parameters
        in a response-time model. Biometrics, 31(4):863–872, 1975.
        https://www.jstor.org/stable/2529811?seq=1#metadata_info_tab_contents

    [3] Michael F. Gensheimer and Balasubramanian Narasimhan. A scalable discrete-time survival model for
        neural networks. PeerJ, 7:e6257, 2019.
        https://peerj.com/articles/6257/
    """
    label_transform = label_transforms.LabTransDiscreteTime

    def __init__(self, net, optimizer=None, device=None, duration_index=None, loss=None):
        self.duration_index = duration_index
        if loss is None:
            loss = models.loss.NLLLogistiHazardLoss()
        super().__init__(net, loss, optimizer, device)

    @property
    def duration_index(self):
        """
        Array of durations that defines the discrete times. This is used to set the index
        of the DataFrame in `predict_surv_df`.

        Returns:
            np.array -- Duration index.
        """
        return self._duration_index

    @duration_index.setter
    def duration_index(self, val):
        self._duration_index = val

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0):
        surv = self.predict_surv(input, batch_size, True, eval_, True, num_workers)
        return pd.DataFrame(surv.transpose(), self.duration_index)

    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                     num_workers=0, epsilon=1e-7):
        hazard = self.predict_hazard(input, batch_size, False, eval_, to_cpu, num_workers)
        surv = (1 - hazard).add(epsilon).log().cumsum(1).exp()
        return tt.utils.array_or_tensor(surv, numpy, input)

    def predict_hazard(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                       num_workers=0):
        hazard = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers).sigmoid()
        return tt.utils.array_or_tensor(hazard, numpy, input)

    def interpolate(self, sub=10, scheme='const_pdf', duration_index=None):
        """Use interpolation for predictions.
        There are two schemes:
            `const_hazard` and `exp_surv` which assumes pice-wise constant hazard in each interval (exponential survival).
            `const_pdf` and `lin_surv` which assumes pice-wise constant PMF in each interval (linear survival).

        Keyword Arguments:
            sub {int} -- Number of "sub" units in interpolation grid. If `sub` is 10 we have a grid with
                10 times the number of grid points than the original `duration_index` (default: {10}).
            scheme {str} -- Type of interpolation {'const_hazard', 'const_pdf'}.
                See `InterpolateDiscrete` (default: {'const_pdf'})
            duration_index {np.array} -- Cuts used for discretization. Does not affect interpolation,
                only for setting index in `predict_surv_df` (default: {None})

        Returns:
            [InterpolateLogisticHazard] -- Object for prediction with interpolation.
        """
        if duration_index is None:
            duration_index = self.duration_index
        return InterpolateLogisticHazard(self, scheme, duration_index, sub)

class SDT(nn.Module):
    """Fast implementation of soft decision tree in PyTorch.

    Parameters
    ----------
    input_dim : int
      The number of input dimensions.
    output_dim : int
      The number of output dimensions. For example, for a multi-class
      classification problem with `K` classes, it is set to `K`.
    depth : int, default=5
      The depth of the soft decision tree. Since the soft decision tree is
      a full binary tree, setting `depth` to a large value will drastically
      increases the training and evaluating cost.
    lamda : float, default=1e-3
      The coefficient of the regularization term in the training loss. Please
      refer to the paper on the formulation of the regularization term.
    use_cuda : bool, default=False
      When set to `True`, use GPU to fit the model. Training a soft decision
      tree using CPU could be faster considering the inherent data forwarding
      process.

    Attributes
    ----------
    internal_node_num_ : int
      The number of internal nodes in the tree. Given the tree depth `d`, it
      equals to :math:`2^d - 1`.
    leaf_node_num_ : int
      The number of leaf nodes in the tree. Given the tree depth `d`, it equals
      to :math:`2^d`.
    penalty_list : list
      A list storing the layer-wise coefficients of the regularization term.
    inner_nodes : torch.nn.Sequential
      A container that simulates all internal nodes in the soft decision tree.
      The sigmoid activation function is concatenated to simulate the
      probabilistic routing mechanism.
    leaf_nodes : torch.nn.Linear
      A `nn.Linear` module that simulates all leaf nodes in the tree.
    """

    def __init__(
            self,
            input_dim,
            output_dim,
            depth=5,
            lamda=1e-3,
            use_cuda=False):
        super(SDT, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.depth = depth
        self.lamda = lamda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self._validate_parameters()

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        # Different penalty coefficients for nodes in different layers
        self.penalty_list = [
            self.lamda * (2 ** (-depth)) for depth in range(0, self.depth)
        ]

        # Initialize internal nodes and leaf nodes, the input dimension on
        # internal nodes is added by 1, serving as the bias.
        self.inner_nodes = nn.Sequential(
            nn.Linear(self.input_dim + 1, self.internal_node_num_, bias=False),
            nn.Sigmoid(),
        )

        self.leaf_nodes = nn.Linear(self.leaf_node_num_,
                                    self.output_dim,
                                    bias=False)

    def forward(self, X, is_training_data=False):
        _mu, _penalty = self._forward(X)
        y_pred = self.leaf_nodes(_mu)

        # When `X` is the training data, the model also returns the penalty
        # to compute the training loss.
        if is_training_data:
            return y_pred, _penalty
        else:
            return y_pred

    def _forward(self, X):
        """Implementation on the data forwarding process."""

        batch_size = X.size()[0]
        X = self._data_augment(X)

        path_prob = self.inner_nodes(X)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

        _mu = X.data.new(batch_size, 1, 1).fill_(1.0)
        _penalty = torch.tensor(0.0).to(self.device)

        # Iterate through internal odes in each layer to compute the final path
        # probabilities and the regularization term.
        begin_idx = 0
        end_idx = 1

        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            # Extract internal nodes in the current layer to compute the
            # regularization term
            _penalty = _penalty + self._cal_penalty(layer_idx, _mu, _path_prob)
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)

            _mu = _mu * _path_prob  # update path probabilities

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = _mu.view(batch_size, self.leaf_node_num_)

        return mu, _penalty

    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        """
        Compute the regularization term for internal nodes in different layers.
        """

        penalty = torch.tensor(0.0).to(self.device)

        batch_size = _mu.size()[0]
        _mu = _mu.view(batch_size, 2 ** layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))

        for node in range(0, 2 ** (layer_idx + 1)):
            alpha = torch.sum(
                _path_prob[:, node] * _mu[:, node // 2], dim=0
            ) / torch.sum(_mu[:, node // 2], dim=0)

            coeff = self.penalty_list[layer_idx]

            penalty -= 0.5 * coeff * (torch.log(alpha) + torch.log(1 - alpha))

        return penalty

    def _data_augment(self, X):
        """Add a constant input `1` onto the front of each sample."""
        batch_size = X.size()[0]
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(self.device)
        X = torch.cat((bias, X), 1)

        return X

    def _validate_parameters(self):

        if not self.depth > 0:
            msg = ("The tree depth should be strictly positive, but got {}"
                   "instead.")
            raise ValueError(msg.format(self.depth))

        if not self.lamda >= 0:
            msg = (
                "The coefficient of the regularization term should not be"
                " negative, but got {} instead."
            )
            raise ValueError(msg.format(self.lamda))