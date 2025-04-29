# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from functools import partial
from typing import Any, Callable, List, Optional, Union

from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import Accuracy, Precision, Recall
from torchmetrics.image import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.text import Perplexity
from torchvision import transforms

from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.evaluation.metrics.utils import metric_data_processor
from pruna.logging.logger import pruna_logger


def default_update(metric: Metric, *args, **kwargs) -> None:
    """
    Default update function for metrics that don't require special handling.

    Parameters
    ----------
    metric : Metric
        The metric instance.
    *args : Any
        The arguments to pass to the metric update method.
    **kwargs : Any
        The keyword arguments to pass to the metric update method.
    """
    metric.update(*args, **kwargs)


# Update functions for metrics that require special handling.
def fid_update(metric: FrechetInceptionDistance, reals: Any, fakes: Any) -> None:
    """
    Update handler for FID metric.

    Parameters
    ----------
    metric : FrechetInceptionDistance instance
        The FID metric instance.
    reals : Any
        The ground truth images tensor.
    fakes : Any
        The generated images tensor.
    """
    metric.update(reals, real=True)
    metric.update(fakes, real=False)


def lpips_update(metric: LearnedPerceptualImagePatchSimilarity, preds: Any, target: Any) -> None:
    """
    Update handler for LPIPS metric.

    Parameters
    ----------
    metric : LearnedPerceptualImagePatchSimilarity instance
        The LPIPS metric instance.
    preds : Any
        The generated images tensor.
    target : Any
        The ground truth images tensor.
    """
    transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])  # converts to [-1, 1]
    preds = preds.float() / 255.0
    target = target.float() / 255.0

    preds = transform(preds)
    target = transform(target)

    metric.update(preds, target)


# Available metrics
class TorchMetrics(Enum):
    """
    Enum for available torchmetrics.

    The enum contains triplets of the metric class, the update function and the call type.

    Parameters
    ----------
    value : Callable
        The function or class constructor for the metric.
    names : List[str]
        The available metric names.
    module : str
        The module in which the metric is defined.
    qualname : str
        Qualified name of the metric.
    type : Type
        The type of the enum value.
    start : int
        The starting value for the enum.
    """

    fid = (partial(FrechetInceptionDistance), fid_update, "gt_y")
    accuracy = (partial(Accuracy), None, "y_gt")
    perplexity = (partial(Perplexity), None, "y_gt")
    clip_score = (partial(CLIPScore), None, "y_x")
    precision = (partial(Precision), None, "y_gt")
    recall = (partial(Recall), None, "y_gt")
    psnr = (partial(PeakSignalNoiseRatio), None, "pairwise_y_gt")
    ssim = (partial(StructuralSimilarityIndexMeasure), None, "pairwise_y_gt")
    lpips = (partial(LearnedPerceptualImagePatchSimilarity), lpips_update, "pairwise_y_gt")

    def __init__(self, *args, **kwargs) -> None:
        self.tm = self.value[0]
        self.update_fn = self.value[1] or default_update
        self.call_type = self.value[2]

    def __call__(self, **kwargs) -> Metric:
        """
        Get an instance of the metric.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments for the metric constructor.

        Returns
        -------
        Metric
            An instance of the torchmetrics metric.
        """
        return self.tm(**kwargs)


@MetricRegistry.register_wrapper(available_metrics=TorchMetrics.__members__.keys())
class TorchMetricWrapper(StatefulMetric):
    """
    Wrapper for torchmetrics.

    Provides a consistent interface for different metrics from torchmetrics.

    Parameters
    ----------
    metric_name : str
        Name of the metric.
    call_type : str
        Specifies the order and type of inputs to use for metric calculation.

        Currently supported formats:
        - 'x_y': Uses input data (x) and model outputs (y)
        - 'gt_y': Uses ground truth (gt) and model outputs (y)
        - 'y_x': Uses model outputs (y) and input data (x)
        - 'y_gt': Uses model outputs (y) and ground truth (gt)

        Future support for pairwise metrics will include additional call_types
        that specify how to compare pairs of outputs for metrics that evaluate
        relationships between outputs of different models.

        This parameter helps determine how the inputs should be arranged
        when calculating the metric.
    **kwargs :
        Additional arguments for the metric constructor.
    """

    def __init__(self, metric_name: str, call_type: str = "", **kwargs) -> None:
        """
        Initialize the torchmetrics metric wrapper.

        Raises
        ------
        ValueError
            If the metric name is not supported.
        """
        super().__init__()
        try:
            if metric_name == "perplexity":
                device = kwargs.pop("device", "cuda")
                self.metric = TorchMetrics[metric_name](**kwargs).to(device)
            else:
                self.metric = TorchMetrics[metric_name](**kwargs)
            # Get the specific update function for the metric, or use the default if not found.
            self.update_fn = TorchMetrics[metric_name].update_fn
        except KeyError:
            raise ValueError(f"Metric {metric_name} is not supported.")

        self.call_type = call_type or TorchMetrics[metric_name].call_type
        if call_type == "pairwise":
            if TorchMetrics[metric_name].call_type.startswith("pairwise"):
                self.call_type = TorchMetrics[metric_name].call_type
            # For some metrics the default call_type is not pairwise.
            # We need to inspect the correct call order from the default call_type
            elif not (TorchMetrics[metric_name].call_type.startswith("y_")):
                self.call_type = "pairwise_gt_y"
            else:
                self.call_type = "pairwise_y_gt"

        pruna_logger.info(f"Using call_type: {self.call_type} for metric {metric_name}")
        self.metric_name = metric_name

    def update(self, x: List[Any] | Tensor, gt: List[Any] | Tensor, outputs: Any) -> None:
        """
        Update the wrapped metric's state with new batch data.

        This method processes the input data through metric_data_processor to arrange inputs
        in the correct order based on the metric's configuration. The arranged inputs are then
        passed to the metric's update function.

        The metric_data_processor supports different input arrangements through 'call_type':

        Parameters
        ----------
        x : List[Any] | Tensor
            The input data.
        gt : List[Any] | Tensor
            The ground truth data.
        outputs : Any
            The output data.
        """
        metric_inputs = metric_data_processor(x, gt, outputs, self.call_type)
        self.update_fn(self.metric, *metric_inputs)

    def add_state(
        self,
        name: str,
        default: Union[list, Tensor],
        dist_reduce_fx: Optional[Union[str, Callable]] = None,
        persistent: bool = False,
    ) -> None:
        """
        Add metric state variables.

        Parameters
        ----------
        name : str
            Name of the state variable.
        default : Union[list, Tensor]
            Default value of the state variable.
        dist_reduce_fx : Optional[Union[str, Callable]], optional
            Function to reduce the state variable in distributed mode.
        persistent : bool, optional
            Whether the state variable should be saved to the ``state_dict`` of the module.
        """
        self.metric.add_state(name, default, dist_reduce_fx, persistent)

    def forward(self, *args, **kwargs) -> None:
        """
        Aggregate and evaluate the batch input directly.

        Parameters
        ----------
        *args : Any
            The arguments to pass to the metric.
        **kwargs : Any
            The keyword arguments to pass to the metric.
        """
        self.metric.forward(*args, **kwargs)

    def reset(self) -> None:
        """Reset the wrapped metric's state."""
        self.metric.reset()

    def compute(self) -> Any:
        """
        Compute the metric value.

        Returns
        -------
        Any
            The computed metric value.
        """
        return self.metric.compute()
