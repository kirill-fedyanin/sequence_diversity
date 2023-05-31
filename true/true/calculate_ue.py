from typing import List

import logging

from true.ue_manager import STRATEGIES

from true.utils import get_method_name_and_strategy_kwargs


log = logging.getLogger()


def calculate_ue(
    ue_methods: List[str],
    output_dict: dict,
    **kwargs,
):
    ue_dict = {}
    for method in ue_methods:
        # If the required inference was successfully generated
        try:
            (
                method_name_upper,
                strategy_kwargs,
            ) = get_method_name_and_strategy_kwargs(
                method=method, kwargs=kwargs, output_dict=output_dict
            )
        except Exception as exception:
            log.warning(f"Could not load data for method {method}: {exception}")
            continue
        log.info(f"Method {method}, strategy {method_name_upper}:")
        scores = STRATEGIES[method_name_upper](**strategy_kwargs, ue_dict=ue_dict)
        if isinstance(scores, dict):
            ue_dict.update(scores)
        else:
            ue_dict[method] = scores
    return ue_dict
