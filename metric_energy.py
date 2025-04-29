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

from typing import Any, Dict

import torch
from codecarbon import EmissionsTracker
from torch.utils.data import DataLoader

from pruna.engine.pruna_model import PrunaModel
from pruna.evaluation.metrics.metric_base import BaseMetric
from pruna.evaluation.metrics.registry import MetricRegistry
from pruna.logging.logger import pruna_logger


@MetricRegistry.register("energy")
class EnergyMetric(BaseMetric):
    """
    Evaluate the energy metrics of a model.

    Measures two key energy performance metrics:
    1. CO2 Emissions: Total carbon emissions produced during inference, measured in kilograms (kg).
    2. Energy Consumption: Total energy consumed during inference, measured in kilowatt-hours (kWh).

    Parameters
    ----------
    n_iterations : int, default=100
        The number of batches to evaluate the model. Note that the energy consumption and CO2 emissions
        are not averaged and will therefore increase with this argument.
    n_warmup_iterations : int, default=10
        The number of warmup batches to evaluate the model.
    device : str | torch.device, default="cuda"
        The device to evaluate the model on.
    """

    def __init__(
        self, n_iterations: int = 100, n_warmup_iterations: int = 10, device: str | torch.device = "cuda"
    ) -> None:
        super().__init__()
        self.n_iterations = n_iterations
        self.n_warmup_iterations = n_warmup_iterations
        self.device = device

    @torch.no_grad()
    def compute(self, model: PrunaModel, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Compute the energy metrics of a model.

        Parameters
        ----------
        model : PrunaModel
            The model to evaluate.
        dataloader : DataLoader
            The dataloader to evaluate the model on.

        Returns
        -------
        dict
            The CO2 emissions and energy consumption of the model.
        """
        # Saving the model to disk to measure loading energy later
        save_path = model.smash_config.cache_dir + "/metrics_save"
        model.save_pretrained(save_path)

        tracker = EmissionsTracker(project_name="pruna", measure_power_secs=0.1)
        tracker.start()

        # Measure the loading energy
        tracker.start_task("Loading model")
        temp_model = model.__class__.from_pretrained(
            save_path,
        )
        tracker.stop_task()
        del temp_model

        model.set_to_eval()
        model.move_to_device(self.device)

        batch = next(iter(dataloader))
        batch = model.inference_handler.move_inputs_to_device(batch, self.device)
        inputs = model.inference_handler.prepare_inputs(batch)

        # Warmup
        for _ in range(self.n_warmup_iterations):
            model(inputs)

        tracker.start_task("Inference")
        for _ in range(self.n_iterations):
            model(inputs)
        tracker.stop_task()

        # Make sure all the operations are finished before stopping the tracker
        if self.device == "cuda" or str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        tracker.stop()

        emissions_data = self._collect_emissions_data(tracker)

        return emissions_data

    def _collect_emissions_data(self, tracker: EmissionsTracker) -> Dict[str, Any]:
        emissions_data = {}
        for task_name, task in tracker._tasks.items():
            emissions_data[f"{task_name}_emissions"] = self._get_data(task.emissions_data, "emissions", task_name)
            emissions_data[f"{task_name}_energy_consumed"] = self._get_data(
                task.emissions_data, "energy_consumed", task_name
            )

        emissions_data["tracker_emissions"] = self._get_data(tracker.final_emissions_data, "emissions", "tracker")
        emissions_data["tracker_energy_consumed"] = self._get_data(
            tracker.final_emissions_data, "energy_consumed", "tracker"
        )

        return emissions_data

    def _get_data(self, source: Any, attribute: str, name: str) -> float:
        try:
            return getattr(source, attribute)
        except AttributeError as e:
            pruna_logger.error(f"Could not get {attribute} data for {name}")
            pruna_logger.error(e)
            return 0
