from typing import Sequence, Any, Dict, List

from src.repositories.db_models import NeuralConfig


class GridSearch:

    def __init__(self):
        pass

    def get_sequence(self, hyper_params: Dict) -> Sequence[NeuralConfig]:
        param_list = list(hyper_params.items())
        sequence = self._get_sequence(param_list)
        config_sequence = [
            {
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "network_size": config["network_shape"][0],
                "depth": config["network_shape"][1],
                "activation_fn": config["activation_fn"]
            }
            for config in sequence
        ]
        return [NeuralConfig.from_dict(config) for config in config_sequence]

    def _get_sequence(self, params_list: List[Any]) -> Sequence[Dict]:
        if len(params_list) == 0:
            return []
        head, *tail = params_list
        name, values = head
        tail_seq = self._get_sequence(tail)
        head_seq = [{name: value} for value in values]
        if len(tail_seq) == 0:
            return head_seq
        result = []
        for head_dict in head_seq:
            for tail_dict in tail_seq:
                result.append(head_dict | tail_dict)
        return result


if __name__ == '__main__':
    search = GridSearch()
    hyper_params = {
        "learning_rate": [1E-6, 3E-7],  # Evenly spaced lr in log scale
        "batch_size": [128, 512],
        "network_size": [2, 4, 6],
        "depth": [2, 4],
        "activation_fn": ["ReLU", "Sigmoid"]
    }
    seq = search.get_sequence(hyper_params)
    print(seq)