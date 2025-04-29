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

import inspect
from types import ModuleType
from typing import Any, List

import diffusers
import diffusers.models.transformers as diffusers_transformers
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
)


def is_causal_lm(model: Any) -> bool:
    """
    Check if the model is a causal LM.

    Parameters
    ----------
    model : Any
        The model to check.

    Returns
    -------
    bool
        True if the model is a causal LM, False otherwise.
    """
    return isinstance(model, tuple(MODEL_FOR_CAUSAL_LM_MAPPING.values()))


def is_translation_model(model: Any) -> bool:
    """
    Check if the model is a translation model.

    Parameters
    ----------
    model : Any
        The model to check.

    Returns
    -------
    bool
        True if the model is a translation model, False otherwise.
    """
    seq2seq_mapping = AutoModelForSeq2SeqLM._model_mapping
    return isinstance(model, tuple(seq2seq_mapping.values()))


def is_speech_seq2seq_model(model: Any) -> bool:
    """
    Check if the model is a speech seq2seq model.

    Parameters
    ----------
    model : Any
        The model to check.

    Returns
    -------
    bool
        True if the model is a speech seq2seq model, False otherwise.
    """
    speech_seq2seq_mapping = AutoModelForSpeechSeq2Seq._model_mapping
    return isinstance(model, tuple(speech_seq2seq_mapping.values()))


def is_diffusers_pipeline(model: Any, include_video: bool = False) -> bool:
    """
    Check if the model is a diffusers pipeline.

    Parameters
    ----------
    model : Any
        The model to check.
    include_video : bool, optional
        Whether to include video pipelines in the check. Default is False.

    Returns
    -------
    bool
        True if the model is a diffusers pipeline (ControlNet, Stable Diffusion,
        Latent Consistency, or optionally video pipeline), False otherwise.
    """
    return (
        is_controlnet_pipeline(model)
        or (is_video_pipeline(model) if include_video else False)
        or is_latent_consistency_pipeline(model)
        or is_sd_pipeline(model)
        or is_sdxl_pipeline(model)
    )


def _check_pipeline_type(model: Any, module_path: ModuleType, prefix: str, suffix: str = "Pipeline") -> bool:
    """
    Generic helper to check pipeline types.

    Parameters
    ----------
    model : Any
        The model to check
    module_path : str
        Path to the diffusers pipeline module
    prefix : str
        Expected prefix for pipeline class names
    suffix : str, optional
        Expected suffix for pipeline class names, defaults to "Pipeline"

    Returns
    -------
    bool
        True if model is an instance of any matching pipeline
    """
    pipelines = dir(module_path)
    pipelines = list(filter(lambda x: x.startswith(prefix) and x.endswith(suffix), pipelines))
    pipeline_classes: list[type] = [getattr(diffusers, p) for p in pipelines]

    return any(isinstance(model, pipeline) for pipeline in pipeline_classes)


def is_controlnet_pipeline(model: Any) -> bool:
    """
    Check if model is a ControlNet pipeline.

    Parameters
    ----------
    model : Any
        The model to check.

    Returns
    -------
    bool
        True if model is a ControlNet pipeline, False otherwise.
    """
    return _check_pipeline_type(model, diffusers.pipelines.controlnet, "StableDiffusion")


def is_video_pipeline(model: Any) -> bool:
    """
    Check if model is a Stable Video Diffusion pipeline.

    Parameters
    ----------
    model : Any
        The model to check.

    Returns
    -------
    bool
        True if model is a Stable Video Diffusion pipeline, False otherwise.
    """
    return _check_pipeline_type(model, diffusers.pipelines.stable_video_diffusion, "StableVideoDiffusion")


def is_latent_consistency_pipeline(model: Any) -> bool:
    """
    Check if model is a Latent Consistency pipeline.

    Parameters
    ----------
    model : Any
        The model to check.

    Returns
    -------
    bool
        True if model is a Latent Consistency pipeline, False otherwise.
    """
    return _check_pipeline_type(model, diffusers.pipelines.latent_consistency_models, "LatentConsistencyModel")


def is_unet_pipeline(model: Any) -> bool:
    """
    Check if model has a diffusers unet as attribute.

    Parameters
    ----------
    model : Any
        The model to check.

    Returns
    -------
    bool
        True if model has a diffusers unet as attribute, False otherwise.
    """
    unet_models = get_diffusers_unet_models()

    for _, attr_value in inspect.getmembers(model):
        if isinstance(attr_value, tuple(unet_models)) and hasattr(model, "unet"):
            return True
    return False


def is_transformer_pipeline(model: Any) -> bool:
    """
    Check if model has a diffusers transformer as attribute.

    Parameters
    ----------
    model : Any
        The model to check.

    Returns
    -------
    bool
        True if model has a diffusers transformer as attribute, False otherwise.
    """
    transformer_models = get_diffusers_transformer_models()

    for _, attr_value in inspect.getmembers(model):
        if isinstance(attr_value, tuple(transformer_models)) and hasattr(model, "transformer"):
            return True
    return False


def is_flux_pipeline(model: Any) -> bool:
    """
    Check if model is a Flux pipeline.

    Parameters
    ----------
    model : Any
        The model to check.

    Returns
    -------
    bool
        True if model is a Flux pipeline, False otherwise.
    """
    return _check_pipeline_type(model, diffusers.pipelines.flux, "Flux")


def is_sdxl_pipeline(model: Any) -> bool:
    """
    Check if model is a Stable Diffusion XL pipeline.

    Parameters
    ----------
    model : Any
        The model to check.

    Returns
    -------
    bool
        True if model is a Stable Diffusion XL pipeline, False otherwise.
    """
    return _check_pipeline_type(model, diffusers.pipelines.stable_diffusion_xl, "StableDiffusionXL")


def is_sd_pipeline(model: Any) -> bool:
    """
    Check if model is a Stable Diffusion pipeline.

    Parameters
    ----------
    model : Any
        The model to check.

    Returns
    -------
    bool
        True if model is a Stable Diffusion pipeline, False otherwise.
    """
    return _check_pipeline_type(model, diffusers.pipelines.stable_diffusion, "StableDiffusion")


def is_sd_3_pipeline(model: Any) -> bool:
    """
    Check if model is a Stable Diffusion 3 pipeline.

    Parameters
    ----------
    model : Any
        The model to check.

    Returns
    -------
    bool
        True if model is a Stable Diffusion 3 pipeline, False otherwise.
    """
    return _check_pipeline_type(model, diffusers.pipelines.stable_diffusion_3, "StableDiffusion3")


def is_hunyuan_pipeline(model: Any) -> bool:
    """
    Check if the model is a Hunyuan pipeline.

    Parameters
    ----------
    model : Any
        The model to check.

    Returns
    -------
    bool
        True if the model is a Hunyuan pipeline, False otherwise.
    """
    return _check_pipeline_type(model, diffusers.pipelines.hunyuan_video, "Hunyuan")


def is_sana_pipeline(model: Any) -> bool:
    """
    Check if the model is a Sana pipeline.

    Parameters
    ----------
    model : Any
        The model to check.

    Returns
    -------
    bool
        True if the model is a Sana pipeline, False otherwise.
    """
    return _check_pipeline_type(model, diffusers.pipelines.sana, "Sana")


def get_helpers(model: Any) -> List[str]:
    """
    Retrieve a list of helper attributes from the model.

    Parameters
    ----------
    model : object
        The model object to inspect.

    Returns
    -------
    List[str]
        A list of attribute names that contain 'helper'.
    """
    return [attr for attr in dir(model) if "helper" in attr and hasattr(model, attr)]


def get_diffusers_transformer_models() -> list:
    """
    Get the transformer models from diffusers.

    Returns
    -------
    list
        The transformer models.
    """
    transformer_models = dir(diffusers_transformers)
    transformer_models = [getattr(diffusers_transformers, x) for x in transformer_models if "Transformer" in x]
    return transformer_models


def get_diffusers_unet_models() -> list:
    """
    Get the unet models from diffusers.

    Returns
    -------
    list
        The unet models.
    """
    unet_models = dir(diffusers.models.unets)
    unet_models = [getattr(diffusers.models.unets, x) for x in unet_models if "UNet" in x]
    return unet_models
