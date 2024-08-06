from typing import Any, Dict

from stable_baselines3.common.policies import BasePolicy as OriginalBasePolicy


class BasePolicy(OriginalBasePolicy):
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(ensemble_size=self.ensemble_size, distribution=self.distribution)
        return data
