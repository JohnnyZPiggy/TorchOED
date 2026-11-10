"""Four rounds of locally linearized D design with a simulated noisy lab."""

from typing import Protocol

import torch

from torchoed import jacobian_features, next_batch


class Backend(Protocol):
    def run(self, x: torch.Tensor) -> torch.Tensor: ...


class SimulatedLab:
    def __init__(self, seed=0):
        self.generator = torch.Generator().manual_seed(seed)

    def run(self, x):
        noise = torch.randn(x.shape, dtype=x.dtype, generator=self.generator)
        return 3.0 * x / (1.2 + x) + 0.05 * noise


class MichaelisMenten(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_theta = torch.nn.Parameter(torch.tensor([2.0, 2.5], dtype=torch.float64).log())

    def forward(self, x):
        V, K = self.log_theta.exp()
        return V * x / (K + x)


def main(seed=0):
    model = MichaelisMenten().eval()
    lab: Backend = SimulatedLab(seed)
    generator = torch.Generator().manual_seed(seed)
    x = torch.linspace(0.1, 10, 100, dtype=torch.float64)
    run_ids, observations, theta = [], [], []
    for round_index in range(4):
        F = jacobian_features(model, x)
        new_ids = next_batch(F, run_ids, 2, criterion="D", generator=generator)
        run_ids.extend(new_ids.tolist())
        observations.append(lab.run(x[new_ids]))
        y = torch.cat(observations)
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = (model(x[run_ids]) - y).square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        estimate = tuple(model.log_theta.detach().exp().tolist())
        theta.append(estimate)
        print(f"round {round_index}: new_ids={new_ids.tolist()}, theta={estimate}")
    return {"run_ids": run_ids, "theta": theta}


if __name__ == "__main__":
    main()
