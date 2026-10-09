# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import os

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse
    from math_verify.grader import verify
    
    # Set the default timeout to 5 seconds
    _TIMEOUT = int(os.environ.get("MATH_VERIFY_TIMEOUT", "5"))
    
    # Monkey patch: modify verify function's default arguments
    # verify.__defaults__ = (float_rounding=6, numeric_precision=15, strict=True, allow_set_relation_comp=False, timeout_seconds=5, raise_on_error=False)
    _verify_defaults = list(verify.__defaults__)
    _verify_defaults[4] = _TIMEOUT  # timeout_seconds at index 5
    verify.__defaults__ = tuple(_verify_defaults)
    
    # Monkey patch: modify parse function's default arguments
    # parse.__defaults__ = (extraction_config, 'first_match', 'any_match', 5, False)
    _parse_defaults = list(parse.__defaults__)
    _parse_defaults[3] = _TIMEOUT  # parsing_timeout at index 3
    parse.__defaults__ = tuple(_parse_defaults)
    
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> dict:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except TimeoutException:
        ret_score = timeout_score
    except Exception:
        pass

    # Convert to consistent dict format like math_dapo.compute_score
    acc = ret_score > 0
    reward = 1.0 if acc else -1.0

    return {
        "score": reward,
        "acc": acc,
    }
