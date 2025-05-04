from typing import Dict, Any, List
import torch


class IntermediateActivationCache:
    __h_space: Dict[float, List[torch.Tensor]]
    """
    keys: time-keys
    values: List of h-space samples for corresponding time key.
    """

    def __init__(self):
        self.__init_cache()
        self.to_cpu = True

    def set_cuda(self):
        self.to_cpu = False

    def add_hspace_sample(self, h_sample: torch.Tensor, t: torch.Tensor) -> None:
        """
        h_sample: shape (bs, *h_shape)
        t: shape (bs,)
        """
        assert len(t.shape) == 1
        assert t.shape[0] == h_sample.shape[0]

        t_cpu = t.detach().cpu()
        if self.to_cpu:
            h_cpu = h_sample.detach().cpu()
            t_keys = t_cpu.numpy().tolist()
            for b_idx, t_key in enumerate(t_keys):
                if t_key not in self.__h_space:
                    self.__h_space[t_key] = []

                self.__h_space[t_key].append(h_cpu[b_idx])
        else:
            h_cpu = h_sample
            t_key = t_cpu.numpy().tolist()[0]
            if t_key not in self.__h_space:
                self.__h_space[t_key] = []
            self.__h_space[t_key].append(h_cpu)

    def time_keys(self) -> List[float]:
        return list(self.__h_space.keys())

    def get_hspace(self, time_key: float) -> List[torch.Tensor] | None:
        if time_key not in self.__h_space:
            return None

        return self.__h_space[time_key]

    def __init_cache(self) -> None:
        self.__h_space = {}

    def clear(self) -> None:
        self.__init_cache()
