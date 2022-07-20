from typing import Sequence, Any, Dict, List

from experiments.Polynomial import Polynomial
from experiments.Experiment import NeuralType
from repositories.db_models import FeedforwardNeuralConfig, ConvolutionalNeuralConfig, NeuralConfig


class GridSearch:

    def __init__(self):
        pass

    def get_sequence(self, hyper_params: Dict, type: NeuralType) -> Sequence[NeuralConfig]:
        if type == "Feedforward":
            return self._get_fnn_sequence(hyper_params)
        elif type == "Convolutional":
            return self._get_cnn_sequence(hyper_params)

    def _get_fnn_sequence(self, hyper_params: Dict) -> Sequence[FeedforwardNeuralConfig]:
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
        return [FeedforwardNeuralConfig.from_dict(config) for config in config_sequence]

    def _get_cnn_sequence(self, hyper_params: Dict) -> Sequence[ConvolutionalNeuralConfig]:
        param_list = list(hyper_params.items())
        sequence = self._get_sequence(param_list)
        config_sequence = [
            {
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "filters": GridSearch._filter_size_from_learnable_params(
                    10 * config["depth"],
                    config["filter_size"],
                    config["learnable_parameters"],
                    config["depth"]
                ),
                "start_size": 10 * config["depth"],
                "filter_size": config["filter_size"],
                "depth": config["depth"],
                "activation_fn": config["activation_fn"]
            }
            for config in sequence
        ]
        return [ConvolutionalNeuralConfig.from_dict(config) for config in config_sequence]

    @staticmethod
    def _filter_size_from_learnable_params(start_size: int, filter_size: int, learnable_params: int, depth: int) -> int:
        polynomial = Polynomial.make_cnn_polynomial(start_size, filter_size, depth)
        root = (polynomial - learnable_params).largest_root()
        return int(round(root))

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
    from experiments.Experiment import Experiment, NeuralType

    search = GridSearch()
    experiment = Experiment("test", "Convolutional")
    seq = search.get_cnn_sequence(experiment.get_hyper_params())
    print(seq)
