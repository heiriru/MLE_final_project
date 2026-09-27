# Double DQN: online selects, target evaluates, no bootstrap after the round ends, and one
# training step changes the zero head.
import numpy as np
import torch

from agent_code.hybrid_agent.hybrid_h3_h8.agent import ResidualQNetwork
from agent_code.hybrid_agent.hybrid_h3_h8.train import double_dqn_targets, residual_total, train_step
from agent_code.hybrid_agent.hybrid_h3_h8.train import Transition


def _biases(model, values):
    with torch.no_grad():
        model.output.bias.copy_(torch.tensor(values, dtype=torch.float32))


def test_double_dqn_uses_online_selection_and_target_evaluation():
    online, target = ResidualQNetwork(), ResidualQNetwork()
    _biases(online, [100., 3., 2., 1., 0., -1.])
    _biases(target, [-50., 7., 11., 13., 17., 19.])
    target_value = double_dqn_targets(
        online, target, torch.tensor([2.]), torch.tensor([0.]), torch.zeros(1, 12), gamma=0.5)
    assert torch.allclose(target_value, torch.tensor([1.875]))


def test_terminal_target_does_not_bootstrap():
    online, target = ResidualQNetwork(), ResidualQNetwork()
    _biases(target, [100.] * 6)
    value = double_dqn_targets(online, target, torch.tensor([-3.]), torch.tensor([1.]),
                              torch.zeros(1, 12), gamma=0.99)
    assert torch.allclose(value, torch.tensor([-3.]))


def test_one_cpu_train_step_updates_the_zero_residual_head():
    online, target = ResidualQNetwork(), ResidualQNetwork()
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    transition = Transition(np.zeros(12, dtype=np.float32), 0, 1.0, np.zeros(12, dtype=np.float32),
                            True)
    loss = train_step(online, target, optimizer, [transition])
    assert loss > 0 and not online.final_layer_is_zero()
    assert residual_total(online, torch.zeros(1, 12)).shape == (1, 6)
