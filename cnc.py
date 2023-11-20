from __future__ import annotations

import numpy as np
from utils import PauliOperator


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
                    beta = pauli.calculate_beta(measured_pauli)
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
                    beta = pauli.calculate_beta(other_pauli)
                    if (gamma_a + gamma_b - gamma_ab) % 2 != beta:
                        return False
        return True

    def __str__(self) -> str:
        return f"CNC State(n={self.n}, omega={self.omega}, gamma={self.gamma})"

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
        initial_state = rng.choice(distribution.keys(), p=distribution.values())
        outcomes = simulate(initial_state, measurements)
        counts.append("".join(str(i) for i in outcomes))

    counts.sort()
    counts = {x: counts.count(x) for x in counts}
    return counts
