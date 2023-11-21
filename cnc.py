from __future__ import annotations

import numpy as np
from utils import PauliOperator
from qutip import Qobj, identity, sigmax, sigmay, sigmaz, tensor
from qutip.measurement import measure
import matplotlib.pyplot as plt
from qiskit.tools.visualization import plot_histogram
import copy

class CNCState:
    def __init__(
        self,
        n: int,
        cnc_set: set[PauliOperator],
        value_assignment: dict[PauliOperator, int],
    ) -> None:
        """
        Initializes the CNCState.

        Parameters:
        n (int): Number of qubits.
        cnc_set (set[PauliOperator]): Set of Pauli operators that are closed under inference and noncontextual.
        value_assignment (dict[int, int]): Noncontextual value assignment for each Pauli operator in the set omega.
                                It takes the Pauli operator as the key and the value is either 0 or 1.
        """
        self._n = n
        if self._check_if_closed(cnc_set):
            self._cnc_set = cnc_set
        else:
            raise RuntimeError(
                "The set of Pauli operators is not closed under inference."
            )

        for value in value_assignment.values():
            if value not in [0, 1]:
                raise RuntimeError(
                    "The value assignment image must be a subset of {0,1}."
                )

        if any([pauli.n != n for pauli in cnc_set]) or any(
            [pauli.n != n for pauli in value_assignment.keys()]
        ):
            raise RuntimeError("The size of the Pauli operators must be n.")

        if set(value_assignment.keys()) != self.cnc_set:
            raise RuntimeError("The value assignment domain must be the cnc set")

        if value_assignment[PauliOperator(np.zeros(2 * n))] != 0:
            raise RuntimeError(
                "The value assignment must be 0 for the identity operator."
            )

        if self._check_if_noncontextual(cnc_set, value_assignment):
            self._value_assignment = value_assignment
        else:
            raise RuntimeError("The set of Pauli operators is not noncontextual.")

    @property
    def n(self) -> int:
        return self._n

    @property
    def cnc_set(self) -> set[PauliOperator]:
        return self._cnc_set

    @property
    def value_assignment(self) -> dict[PauliOperator, int]:
        return self._value_assignment

    def update(self, measured_pauli: PauliOperator) -> int:
        """
        Updates the CNC state according to given measurement, and returns the outcome of the measurement.

        Parameters:
        measured_pauli (PauliOperator): Pauli operator that is measured.

        Returns:
        int: Outcome of the measurement.
        """
        rng = np.random.default_rng()
        if measured_pauli in self.cnc_set:
            outcome = self.value_assignment[measured_pauli]
            if rng.choice([0, 1]) == 1:
                for pauli in self.value_assignment:
                    omega = measured_pauli.calculate_omega(pauli)
                    self._value_assignment[pauli] = (
                        self.value_assignment[pauli] + omega
                    ) % 2
        else:
            outcome = rng.choice([0, 1])
            commuting_paulis = {
                pauli
                for pauli in self.cnc_set
                if pauli.calculate_omega(measured_pauli) == 0
            }
            added_set = {measured_pauli + pauli for pauli in commuting_paulis}
            self._cnc_set = commuting_paulis.union(added_set)
            new_value_assignment = {}
            for pauli in self.cnc_set:
                if pauli in commuting_paulis:
                    new_value_assignment[pauli] = self.value_assignment[pauli]
                else:
                    beta = pauli.calculate_beta(measured_pauli) % 2
                    new_value_assignment[pauli] = (
                        self.value_assignment[pauli + measured_pauli] + outcome + beta
                    ) % 2
            self._value_assignment = new_value_assignment
        return outcome

    def _check_if_closed(self, cnc_set: set[PauliOperator]) -> bool:
        """
        Checks if the set of Pauli operators is closed under inference.

        Parameters:
        cnc_set (set[PauliOperator]): Set of Pauli operators.

        Returns:
        bool: True if the set is closed under inference and noncontextual, False otherwise.
        """
        for pauli in cnc_set:
            for other_pauli in cnc_set:
                if (
                    pauli.calculate_omega(other_pauli) == 0
                    and pauli + other_pauli not in cnc_set
                ):
                    return False
        return True

    def _check_if_noncontextual(
        self, cnc_set: set[PauliOperator], value_assignment: dict[PauliOperator, int]
    ) -> bool:
        """
        Checks if the set of Pauli operators is noncontextual.

        Parameters:
        cnc_set (set[np.ndarray]): Set of Pauli operators.
        value_assignment (dict[int, int]): Noncontextual value assignment for each Pauli operator in the set omega.
                                It takes the order of the Pauli operator as the key and the value is either 0 or 1.

        Returns:
        bool: True if the set is closed under inference and noncontextual, False otherwise.
        """
        for pauli in cnc_set:
            for other_pauli in cnc_set:
                if pauli.calculate_omega(other_pauli) == 0:
                    gamma_a = value_assignment[pauli]
                    gamma_b = value_assignment[other_pauli]
                    gamma_ab = value_assignment[(pauli + other_pauli)]
                    beta = pauli.calculate_beta(other_pauli) % 2
                    if (gamma_a + gamma_b - gamma_ab) % 2 != beta:
                        return False
        return True

    def __str__(self) -> str:
        return f"CNC State(n={self.n}, cnc_set={self.cnc_set}, value_assignment={self.value_assignment})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: CNCState) -> bool:
        return (
            self.n == other.n
            and self.cnc_set == other.cnc_set
            and self.value_assignment == other.value_assignment
        )

    def __hash__(self) -> int:
        return hash(
            (self.n, frozenset(self.cnc_set), frozenset(self.value_assignment.items()))
        )


def simulate(initial_state: CNCState, measurements: list[PauliOperator]) -> list[int]:
    """
    Classical simulation algorithm with given measurements after the initial state is sampled.

    Parameters:
    initial_state (CNCState): Initial CNC state.
    measurements (list[PauliOperator]): List of Pauli operators that are going to be measured.

    Returns:
    list[int]: Outcomes of the measurements.
    """
    state = initial_state
    outcomes = []
    for measurement in measurements:
        outcome = state.update(measurement)
        outcomes.append(outcome)

    return outcomes


def run_simulations_with_distribution(
    distribution: dict[PauliOperator, float],
    measurements: list[PauliOperator],
    num_simulations: int,
) -> dict[str, int]:
    """
    Run simulations on given distribution and measurements as many times as num_simulations.

    Parameters:
    distribution (dict[PauliOperator, float]): Distribution over the set of Pauli operators. Each pair is a Pauli operator and its probability.
    measurements (list[PauliOperator]): List of Pauli operators that are going to be measured.
    num_simulations (int): Number of simulations.

    Returns:
    dict[str, int]: Dictionary of outcomes and their counts.
    """
    rng = np.random.default_rng()
    counts = []
    for _ in range(num_simulations):
        initial_state = rng.choice(
            list(distribution.keys()), p=list(distribution.values())
        )
        initial_state = copy.deepcopy(initial_state)
        outcomes = simulate(initial_state, measurements)
        counts.append("".join(str(i) for i in outcomes))

    counts.sort()
    counts = {x: counts.count(x) for x in counts}
    return counts

def qutip_simuation(
    initial_state: Qobj, measurements: list[Qobj], num_simulations: int
) -> dict[str, int]:
    counts = []
    for _ in range(num_simulations):
        state = initial_state
        outcomes = []
        for measurement in measurements:
            outcome, state = measure(state, measurement)
            if outcome == 1:
                outcome = 0
            else:
                outcome = 1
            outcomes.append(outcome)

        counts.append("".join(str(i) for i in outcomes))

    counts.sort()
    counts = {x: counts.count(x) for x in counts}

    return counts

if __name__ == "__main__":
    num_simulations = 4096
    cnc_set = {
        PauliOperator('II'),
        PauliOperator('IZ'),
        PauliOperator('ZI'),
        PauliOperator('ZZ'),
        PauliOperator('YI'),
        PauliOperator('YZ'),
        PauliOperator('XI'),
        PauliOperator('XZ')
    }

    value_assignment1 = {
        pauli: 0 for pauli in cnc_set
    }

    value_assignment2 = {
        PauliOperator('II'): 0,
        PauliOperator('IZ'): 0,
        PauliOperator('ZI'): 0,
        PauliOperator('ZZ'): 0,
        PauliOperator('YI'): 1,
        PauliOperator('YZ'): 1,
        PauliOperator('XI'): 1,
        PauliOperator('XZ'): 1
    }

    initial_state1 = CNCState(2, cnc_set, value_assignment1)
    initial_state2 = CNCState(2, cnc_set, value_assignment2)

    measurements = [
        PauliOperator('YY'),
        PauliOperator('ZI'),
    ]

    distribution = {
        initial_state1: 0.5,
        initial_state2: 0.5
    }

    counts_cnc = run_simulations_with_distribution(distribution, measurements, num_simulations)

    II = tensor(identity(2), identity(2))
    ZI = tensor(sigmaz(), identity(2))
    IZ = tensor(identity(2), sigmaz())
    ZZ = tensor(sigmaz(), sigmaz())
    YY = tensor(sigmay(), sigmay())

    rho = 1/4 * (II + ZI + IZ + ZZ)

    measurements = [YY, ZI]

    counts_qutip = qutip_simuation(rho, measurements, num_simulations)

    colors = ["blue", "orange"]

    plot_histogram([counts_qutip, counts_cnc], color=colors)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    labels = ["Qutip", "CNC"]
    plt.legend(handles, labels)
    plt.title("rho in YY, ZI Bases")
    plt.show()
