# coding=utf-8
# Copyright 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for computing gradient alignment statistics."""

from typing import Any

import jax
import jax.numpy as jnp


def cosine_alignment(grad_a: Any, grad_b: Any, eps: float = 1e-8) -> jnp.ndarray:
    """Return cosine similarity between two gradient pytrees.

    Args:
      grad_a: First gradient pytree.
      grad_b: Second gradient pytree.
      eps: Small constant to avoid division by zero.

    Returns:
      Scalar cosine similarity between grad_a and grad_b.
    """
    flat_a, _ = jax.flatten_util.ravel_pytree(grad_a)
    flat_b, _ = jax.flatten_util.ravel_pytree(grad_b)
    dot = jnp.vdot(flat_a, flat_b)
    norm_a = jnp.linalg.norm(flat_a)
    norm_b = jnp.linalg.norm(flat_b)
    return dot / (norm_a * norm_b + eps)

