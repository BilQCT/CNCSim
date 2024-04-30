from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
from qiskit.tools.visualization import plot_histogram
from qutip import identity, sigmax, sigmaz, tensor

from utils import (H_GATE, T_GATE, Pauli, binary_array_to_decimal,
                   decimal_to_binary_array, qutip_simuation)


class HyperCubeManager:
    def __init__(self, n: int) -> None:
        self._n = n
        self._commuting_paulis = self.find_commuting_paulis(n)

    @property
    def n(self) -> int:
        return self._n

    @property
    def commuting_paulis(self) -> dict[Pauli, list[Pauli]]:
        return self._commuting_paulis

    def find_pauli_coefficients(self, rho: np.ndarray) -> np.ndarray:
        alphas = np.zeros(2 ** (2 * self.n))
        for i in range(2 ** (2 * self.n)):
            pauli_i = Pauli(decimal_to_binary_array(i, width=2 * self.n)).get_operator()
            alphas[i] = np.trace(pauli_i @ rho).real

        return alphas

    def sample_initial_state(self, alphas: np.ndarray) -> np.ndarray:
        f_values = [0]
        rng = np.random.default_rng()
        for i in range(1, len(alphas)):
            alpha = alphas[i]
            zero_prob = (1 + alpha) / 2
            one_prob = (1 - alpha) / 2
            prob_dist = [zero_prob, one_prob]
            f_value = rng.choice(2, p=prob_dist)
            f_values.append(f_value)

        return np.array(f_values)

    def find_commuting_paulis(self, n: int) -> dict[np.ndarray, list[np.ndarray]]:
        paulis = [Pauli(np.array(p)) for p in itertools.product([0, 1], repeat=2 * n)]
        result = {paulis[i]: [paulis[i]] for i in range(len(paulis))}
        for i in range(len(paulis)):
            for j in range(i + 1, len(paulis)):
                if paulis[i].calculate_omega(paulis[j]) == 0:
                    result[paulis[i]].append(paulis[j])
                    result[paulis[j]].append(paulis[i])
        return result

    def find_transition_state(self, state: np.ndarray, pauli: Pauli) -> np.ndarray:
        index = pauli.associated_integer
        s = state[index]
        result = ((state + np.ones(len(state))) % 2).astype(int)
        for comm_pauli in self.commuting_paulis[pauli]:
            comm_index = comm_pauli.associated_integer
            gamma = comm_pauli.calculate_beta(pauli) % 2

            # Note that addition in the formula is defined in arrays and in modulo 2 not in their order
            added_index = binary_array_to_decimal((pauli.bsf + comm_pauli.bsf) % 2)
            result[comm_index] = (s + gamma + state[added_index]) % 2

        return result

    def simulate(
        self, initial_state: np.ndarray, measurements: list[Pauli]
    ) -> list[int]:
        state = initial_state
        rng = np.random.default_rng()
        outcomes = []
        for measurement in measurements:
            index = measurement.associated_integer
            s = int(state[index])
            # If the coin flip is 1, then we transition to a new state, otherwise we stay in the same state
            if rng.choice([0, 1]) == 1:
                state = self.find_transition_state(state, measurement)
            outcomes.append(s)

        return outcomes

    def run_simulations_with_state(
        self,
        rho: np.ndarray,
        measurements: list[Pauli],
        num_simulations: int,
    ) -> dict[str, int]:
        counts = []
        alphas = self.find_pauli_coefficients(rho)
        for _ in range(num_simulations):
            initial_state = self.sample_initial_state(alphas)
            outcomes = self.simulate(initial_state, measurements)
            counts.append("".join(str(i) for i in outcomes))

        counts.sort()
        counts = {x: counts.count(x) for x in counts}
        return counts

    def run_simulations_with_distribution(
        self,
        distribution: dict[tuple, float],
        measurements: list[Pauli],
        num_simulations: int,
    ) -> dict[str, int]:
        rng = np.random.default_rng()
        counts = []
        for _ in range(num_simulations):
            initial_state = rng.choice(
                list(distribution.keys()), p=list(distribution.values())
            )
            outcomes = self.simulate(initial_state, measurements)
            counts.append("".join(str(i) for i in outcomes))

        counts.sort()
        counts = {x: counts.count(x) for x in counts}
        return counts


def one_qubit_example():
    n = 1
    hm = HyperCubeManager(n)
    rho = T_GATE @ H_GATE @ np.array([[1, 0], [0, 0]]) @ H_GATE @ T_GATE.conj().T
    measurements = [
        Pauli(np.array([1, 0])),
        Pauli(np.array([1, 0])),
        Pauli(np.array([1, 1])),
        Pauli(np.array([0, 1])),
    ]
    counts = hm.run_simulations_with_state(rho, measurements, 2048)
    plot_histogram(counts)
    plt.show()


def two_qubit_counter_example():
    n = 2
    num_simulations = 4096
    hm = HyperCubeManager(n)

    # Create the distribution of vertices as in the written example
    a_f = np.zeros(16)
    a_fbar = np.ones(16)
    a_fbar[0] = 0
    a_f = tuple(a_f)
    a_fbar = tuple(a_fbar)
    distribution = {a_f: 0.5, a_fbar: 0.5}

    # Z1, ZX
    measurements = [
        Pauli("ZI"),
        Pauli("ZX"),
    ]
    counts_hm = hm.run_simulations_with_distribution(
        distribution, measurements, num_simulations
    )

    # Define the Pauli matrices
    Z1 = tensor(sigmaz(), identity(2))
    ZX = tensor(sigmaz(), sigmax())

    # Initialize the state as the maximally mixed state
    state = tensor(identity(2), identity(2)) / 4

    # Define the measurement operators
    measurements = [Z1, ZX]

    counts_qutip = qutip_simuation(state, measurements, num_simulations)

    colors = ["blue", "orange"]

    plot_histogram([counts_qutip, counts_hm], color=colors)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    labels = ["Qutip", "Hypercube"]
    plt.legend(handles, labels)
    plt.title("Maximally mixed state measured in Z1, ZX Bases")
    plt.show()


if __name__ == "__main__":
    two_qubit_counter_example()
