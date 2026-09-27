# Three-step returns and the flush at the end of a round; n=1 must give exactly the one-step
# target.
import numpy as np

from agent_code.hybrid_agent.hybrid_h3_h8.train import NStepAccumulator
from agent_code.hybrid_agent.hybrid_h3_h8.train import Transition


def item(reward, done=False):
    return Transition(np.zeros(12), 0, reward, np.ones(12), done, np.ones(6, dtype=bool))


def test_three_step_return_and_terminal_flush():
    queue = NStepAccumulator(n=3, gamma=0.5)
    assert queue.append(item(1.0)) == []
    assert queue.append(item(2.0)) == []
    first = queue.append(item(4.0))
    assert len(first) == 1 and first[0].reward == 3.0 and first[0].n_steps == 3
    tail = queue.append(item(8.0, done=True))
    assert [round(value.reward, 3) for value in tail] == [6.0, 8.0, 8.0]
    assert [value.n_steps for value in tail] == [3, 2, 1]


def test_one_step_is_the_h3_identity_case():
    queue = NStepAccumulator(n=1, gamma=0.99)
    output = queue.append(item(2.5))
    assert len(output) == 1 and output[0].reward == 2.5 and output[0].n_steps == 1
