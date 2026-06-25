# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional, TypedDict

import pytest

from tensorrt_llm._torch.models.modeling_multimodal_utils import _get_cached_merged_typed_dict


def test_get_cached_merged_typed_dict_reuses_schema():
    import huggingface_hub.dataclasses as hf_dataclasses

    cache = {}
    first_schema = TypedDict("merged_typed_dict", {"do_rescale": Optional[bool]}, total=False)
    second_schema = TypedDict("merged_typed_dict", {"do_rescale": Optional[bool]}, total=False)

    first_cached_schema = _get_cached_merged_typed_dict(first_schema, cache)
    second_cached_schema = _get_cached_merged_typed_dict(second_schema, cache)

    assert first_cached_schema is second_cached_schema
    hf_dataclasses.validate_typed_dict(first_cached_schema, {"do_rescale": False})
    hf_dataclasses.validate_typed_dict(second_cached_schema, {"do_rescale": True})

    with pytest.raises(TypeError):
        hf_dataclasses.validate_typed_dict(second_cached_schema, {"not_a_real_kwarg": 1})


def test_get_cached_merged_typed_dict_ignores_other_schemas():
    other_schema = TypedDict("other_typed_dict", {"do_rescale": Optional[bool]}, total=False)

    assert _get_cached_merged_typed_dict(other_schema, {}) is other_schema
