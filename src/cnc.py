from __future__ import annotations

import numpy as np

from src.utils import (Pauli, PhasePointOperator,
                       get_n_from_pauli_basis_representation)

from copy import deepcopy

class CNC(PhasePointOperator):
    """Class for the Closed-Under-Inference and Noncontextual (CNC) operators.

    Note:
        The CNC operator is defined using a set of Pauli operators which we call :py:attr:`omega` and a function
        from this set to {0, 1} which we call :py:attr:`gamma`. The set :py:attr:`omega` is closed under inference
        and the function :py:attr:`gamma` is noncontextual. The CNC operator is defined as the pair
        (:py:attr:`omega`, :py:attr:`gamma`). We can write the n qubit CNC operator as,

        .. math::
            A_{\\Omega}^{\\gamma} = \\frac{1}{2^n} \\sum_{a \\in \\Omega} (-1)^{\\gamma(a)} T_a

        For detailed information, see https://arxiv.org/abs/1905.05374.
    """

    def __init__(
        self,
        gamma: dict[Pauli, int],
    ) -> None:
        """Initializes the CNC operator.

        Args:
            gamma (dict[Pauli, int]): Noncontextual value assignment for each Pauli operator in the set omega.
                It takes the Pauli operator as the key and the value is either 0 or 1.
        """

        # Check whether the Pauli operators have the same size.
        n = list(gamma.keys())[0].n
        if any([pauli.n != n for pauli in gamma.keys()]):
            raise ValueError("All Pauli operators must have the same size.")
        pass

        # Check whether given set of Pauli operators is closed under inference.
        for pauli in gamma.keys():
            for other_pauli in gamma.keys():
                if (
                    pauli.calculate_omega(other_pauli) == 0
                    and pauli + other_pauli not in gamma.keys()
                ):
                    raise ValueError(
                        "Given set of Pauli operators is not closed under inference."
                    )

        # Check whether gamma has values in {0, 1}.
        for value in gamma.values():
            if value not in [0, 1]:
                raise ValueError(
                    "The value assignment image must be a subset of {0,1}."
                )

        # Check whether the identity operator is in the value assignment.
        identity = Pauli.identity(n)
        if identity not in gamma.keys():
            raise ValueError("The identity operator must be in the value assignment.")

        # Check whether the value assignment is 0 for the identity operator.
        if gamma[identity] != 0:
            raise ValueError(
                "The value assignment must be 0 for the identity operator."
            )

        # Check whether given gamma is noncontextual.
        for pauli in gamma.keys():
            for other_pauli in gamma.keys():
                if pauli.calculate_omega(other_pauli) == 0:
                    gamma_a = gamma[pauli]
                    gamma_b = gamma[other_pauli]
                    gamma_ab = gamma[(pauli + other_pauli)]
                    beta = pauli.calculate_beta(other_pauli)
                    if (gamma_a + gamma_b - gamma_ab) % 2 != beta:
                        raise ValueError("Given value assignment is not noncontextual.")

        self._n = n
        self._gamma = deepcopy(gamma)
        self._omega = set(gamma.keys())

    @property
    def n(self) -> int:
        """Number of qubits."""
        return self._n

    @property
    def omega(self) -> set[Pauli]:
        """Set of Pauli operators in the CNC. By definition of CNCs, this set is closed under inference."""
        return self._omega

    @property
    def gamma(self) -> dict[Pauli, int]:
        """Noncontextual value assignment for each Pauli operator in the set omega. It takes the Pauli operator
        as the key and the value is either 0 or 1."""
        return self._gamma

    @classmethod
    def from_pauli_basis_representation(cls, basis_representation: np.ndarray) -> CNC:
        """Creates a CNC operator from a Pauli basis representation.

        Args:
            n (int): Number of qubits.
            basis_representation (np.ndarray): Pauli basis representation of the CNC operator. In a Pauli basis
                representation, value of the index i is the value of the :py:attr:`gamma` for the Pauli with
                :py:attr:`~utils.Pauli.basis_order` i.

        Returns:
            CNC: CNC operator created from the Pauli basis representation.
        """
        n = get_n_from_pauli_basis_representation(basis_representation)
        gamma = {}
        for i, value in enumerate(basis_representation):
            pauli = Pauli.from_basis_order(n, i)
            if value == 1:
                gamma[pauli] = 0
            elif value == -1:
                gamma[pauli] = 1

        return cls(gamma)

    def get_pauli_basis_representation(self) -> np.ndarray:
        """Returns the Pauli basis representation of the CNC operator. In a Pauli basis representation,value of
        the index i is the value of the :py:attr:`gamma` for the Pauli with :py:attr:`~utils.Pauli.basis_order` i.

        Returns:
            np.ndarray: Pauli basis representation of the CNC operator.
        """
        basis_representation = np.zeros(4**self.n, dtype=int)
        for pauli, value in self.gamma.items():
            index = pauli.basis_order
            basis_representation[index] = (int)(-1) ** value

        return basis_representation

    def update(self, measured_pauli: Pauli) -> int:
        """Updates the CNC state according to given measurement, and returns the outcome of the measurement.
        For detailed information, see https://arxiv.org/abs/1905.05374.

        Args:
            measured_pauli (Pauli): Pauli operator that is measured.

        Returns:
            int: Outcome of the measurement.
        """
        rng = np.random.default_rng()
        if measured_pauli in self.gamma.keys():
            # If the measured Pauli operator is in the set omega
            outcome = self.gamma[measured_pauli]
            if rng.choice([0, 1]) == 1:
                for pauli in self.gamma:
                    omega = measured_pauli.calculate_omega(pauli)
                    self._gamma[pauli] = (self.gamma[pauli] + omega) % 2
        else:
            # If the measured Pauli operator is not in the set omega
            outcome = rng.choice([0, 1])
            commuting_paulis = {
                pauli
                for pauli in self.omega
                if pauli.calculate_omega(measured_pauli) == 0
            }
            poset_paulis = {measured_pauli + pauli for pauli in commuting_paulis}
            new_omega = commuting_paulis.union(poset_paulis)
            new_gamma = {}
            for pauli in new_omega:
                if pauli in commuting_paulis:
                    new_gamma[pauli] = self.gamma[pauli]
                else:
                    beta = pauli.calculate_beta(measured_pauli) % 2
                    new_gamma[pauli] = (
                        self.gamma[pauli + measured_pauli] + outcome + beta
                    ) % 2

            self._gamma = new_gamma
            self._omega = new_omega
        return outcome

    def __str__(self) -> str:
        return f"CNC(n={self.n} with {len(self.omega)} Pauli operators)"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: CNC) -> bool:
        return self.gamma == other.gamma

    def __hash__(self) -> int:
        return hash(frozenset(self.gamma.items()))


# TODO : Create MaximalCNC class


def simulate_from_sampled_cnc(sampled_cnc: CNC, measurements: list[Pauli]) -> list[int]:
    """Classical simulation algorithm with given measurements after the initial state is sampled. For detailed information,
    see https://arxiv.org/abs/1905.05374.

    Args:
        sampled_cnc (CNC): Sampled CNC operator.
        measurements (list[Pauli]): List of Pauli operators that are going to be measured.

    Returns:
        list[int]: Outcomes of the measurements.
    """
    cnc = sampled_cnc
    outcomes = []
    for measurement in measurements:
        outcome = cnc.update(measurement)
        outcomes.append(outcome)
    return outcomes

import copy

def simulate_from_distribution(
    dist: dict[CNC, float],
    measurements: list[Pauli],
    num_simulations: int = 1024,
) -> dict[str, int]:
    """Run simulations on given distribution and measurements as many times as num_simulations. For detailed information,
    see https://arxiv.org/abs/1905.05374.

    Args:
        dist (dict[CNC, float]): Distribution over the set of CNC operators. Each pair is a CNC operator and its probability.
        measurements (list[Pauli]): List of Pauli operators that are going to be measured.
        num_simulations (int): Number of simulations.

    Returns:
        dict[str, int]: Dictionary of outcomes and their counts.
    """
    rng = np.random.default_rng()
    counts = []
    for _ in range(num_simulations):
        sampling_dist = copy.deepcopy(dist)
        initial_state = rng.choice(list(sampling_dist.keys()), p=list(sampling_dist.values()))
        outcomes = simulate_from_sampled_cnc(initial_state, measurements)
        counts.append("".join(str(i) for i in outcomes))
    counts.sort()
    counts = {x: counts.count(x) for x in counts}
    return counts
