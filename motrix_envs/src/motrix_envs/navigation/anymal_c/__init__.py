# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
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
# ==============================================================================

"""Anymal C Navigation Environment module.

This module provides:
- AnymalCNavEnvCfg: Configuration for the Anymal C navigation environment
- AnymalCNavTask: The navigation task environment implementation

Registered environment:
- "anymal-c-flat-terrain-nav": Navigation task on flat terrain
"""

from motrix_envs.navigation.anymal_c.cfg import AnymalCNavEnvCfg
from motrix_envs.navigation.anymal_c.anymal_c_np import AnymalCNavTask

__all__ = ["AnymalCNavEnvCfg", "AnymalCNavTask"]
