from __future__ import annotations

import copy
import logging
from typing import Optional

import galois
import numpy as np

GF2 = galois.GF(2)

'''
def symplectic_inner_product(u: np.ndarray, v: np.ndarray) -> int:
    """Computes the symplectic inner product of vectors u and v over GF(2)."""
    n = len(u) // 2
    ux, uz = u[:n], u[n:]
    vx, vz = v[:n], v[n:]
    return (uz.dot(vx) - ux.dot(vz)) % 2
''';
    

def symplectic_inner_product(u: np.ndarray, v: np.ndarray) -> int:
    """Computes the symplectic inner product of vectors u and v over GF(2)."""
    n = len(u) // 2
    ux, uz = u[:n].astype(np.int64), u[n:].astype(np.int64)
    vx, vz = v[:n].astype(np.int64), v[n:].astype(np.int64)
    return (uz.dot(vx) - ux.dot(vz)) % 2



def beta(
    u: np.ndarray, v: np.ndarray, skip_commutation_check: bool = False
) -> int:
    n = len(u) // 2

    if not skip_commutation_check:
        assert symplectic_inner_product(u, v) == 0

    ux, uz = u[:n], u[n:]
    vx, vz = v[:n], v[n:]

    x_terms = (ux ^ vx) % 2
    z_terms = (uz ^ vz) % 2

    u_phase = ux.dot(uz) % 4
    v_phase = vx.dot(vz) % 4
    combined_phase = x_terms.dot(z_terms) % 4

    #gamma = (u_phase + v_phase + 2 * uz.dot(vx) - combined_phase) % 4
    gamma = (
    u_phase.astype(np.int32)
    + v_phase.astype(np.int32)
    + 2 * uz.astype(np.int32).dot(vx.astype(np.int32))
    - combined_phase.astype(np.int32)
) % 4

    beta = gamma // 2

    return beta


def symplectic_matrix(n: int) -> np.ndarray:
    zeros = np.zeros((n, n), dtype=np.uint8)
    identity = np.eye(n, dtype=np.uint8)
    return np.block([[zeros, identity], [identity, zeros]])


def pauli_bsf_to_str(pauli_bsf: np.ndarray) -> str:
    """Converts Binary Symplectic Form of a Pauli operator to its string
    representation.
    For example [1, 0, 0, 1] is converted to "XZ".

    Args:
        pauli_bsf (np.ndarray): Binary Symplectic Form of the Pauli operator.

    Returns:
        str: String representation of the Pauli operator.
    """
    if not isinstance(pauli_bsf, np.ndarray):
        raise TypeError(
            "Binary symplectic form of the Pauli operator "
            "must be a numpy array."
        )

    if not all(i in [0, 1] for i in pauli_bsf):
        raise ValueError(
            "Binary symplectic form of the Pauli operator is not "
            "a binary array."
        )

    if len(pauli_bsf) % 2 == 1 or len(pauli_bsf) <= 0:
        raise ValueError(
            "Size of the given Binary Symplectic Form is not correct. "
            "It must be a positive even number."
        )

    pauli_str = ""

    n = len(pauli_bsf) // 2
    for i in range(n):
        if pauli_bsf[i] == 1 and pauli_bsf[i + n] == 1:
            pauli_str += "Y"
        elif pauli_bsf[i] == 1:
            pauli_str += "X"
        elif pauli_bsf[i + n] == 1:
            pauli_str += "Z"
        else:
            pauli_str += "I"

    return pauli_str


class CncSimulator:
    def __init__(self, n: int, m: int, seed: Optional[int] = None) -> None:
        logging.debug(
            f"Initializing CncSimulator with n={n}, m={m}, seed={seed}"
        )
        self._n = n
        self._m = m
        self._isotropic_dim = n - m
        self._symplectic_matrix = symplectic_matrix(n)
        self._tableau = self.initial_cnc_tableau(n, m)
        self._tableau_without_phase = self._tableau[:, :-1]
        self._x_cols = self._tableau[:, :n]
        self._z_cols = self._tableau[:, n:-1]
        self._phase_col = self._tableau[:, -1]
        self._destabilizer_rows = self._tableau_without_phase[
            : self.isotropic_dim
        ]
        self._stabilizer_rows = self._tableau_without_phase[
            self.isotropic_dim: 2 * self.isotropic_dim
        ]
        self._jw_elements_rows = self._tableau_without_phase[
            2 * self.isotropic_dim:
        ]

        # Initialize local random generator
        self._rng = np.random.default_rng(seed)

    def __deepcopy__(self, memo: Optional[dict] = None) -> CncSimulator:
        new_instance = CncSimulator(self.n, self.m)
        new_instance._tableau = copy.deepcopy(self._tableau)
        new_instance._tableau_without_phase = new_instance._tableau[:, :-1]
        new_instance._x_cols = new_instance._tableau[:, :self.n]
        new_instance._z_cols = new_instance._tableau[:, self.n:-1]
        new_instance._phase_col = new_instance._tableau[:, -1]
        new_instance._destabilizer_rows = new_instance._tableau_without_phase[
            : self.isotropic_dim
        ]
        new_instance._stabilizer_rows = new_instance._tableau_without_phase[
            self.isotropic_dim: 2 * self.isotropic_dim
        ]
        new_instance._jw_elements_rows = new_instance._tableau_without_phase[
            2 * self.isotropic_dim:
        ]
        return new_instance

    def __eq__(self, other: CncSimulator) -> bool:
        if not isinstance(other, CncSimulator):
            return NotImplemented

        return (
            self._n == other._n and
            self._m == other._m and
            np.array_equal(self._tableau, other._tableau)
        )

    def __str__(self) -> str:
        title = f"CNC Simulator with n={self.n}, m={self.m}"

        # Convert each destabilizer row to a string representation
        destabilizer_strings = [
            pauli_bsf_to_str(row) for row in self._destabilizer_rows
        ]
        # Join all destabilizer strings with commas and enclose them
        # within angle brackets
        formatted_destabilizers = (
            f"Destabilizers: <{', '.join(destabilizer_strings)}>"
        )

        # Convert each stabilizer string to a string representation
        stabilizer_strings = [
            pauli_bsf_to_str(row) for row in self._stabilizer_rows
        ]
        # Add minus if the stabilizer has a negative phase
        stabilizer_strings = [
            f"-{s}" if self._phase_col[self.isotropic_dim + i] == 1 else s
            for i, s in enumerate(stabilizer_strings)
        ]
        # Join all stabilizer strings with commas and enclose them
        # within angle brackets
        formatted_stabilizers = (
            f"Stabilizers: <{', '.join(stabilizer_strings)}>"
        )

        # Convert each JW element to a string representation
        jw_element_strings = [
            pauli_bsf_to_str(row) for row in self._jw_elements_rows
        ]
        # Add minus if the JW element has a negative phase
        jw_element_strings = [
            f"-{s}" if self._phase_col[2 * self.isotropic_dim + i] == 1 else s
            for i, s in enumerate(jw_element_strings)
        ]
        # Join all JW element strings with commas and enclose them
        formatted_jw_elements = (
            f"JW Elements: {', '.join(jw_element_strings)}"
        )

        return "\n".join(
            [
                title,
                formatted_destabilizers,
                formatted_stabilizers,
                formatted_jw_elements
            ]
        )

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def from_tableau(cls, n: int, m: int, tableau: np.ndarray) -> CncSimulator:
        if not cls.is_cnc(n, m, tableau):
            raise ValueError("Given tableau is not a valid CNC tableau.")

        instance = cls(n, m)
        instance._tableau = copy.deepcopy(tableau)
        instance._tableau_without_phase = instance._tableau[:, :-1]
        instance._x_cols = instance._tableau[:, :n]
        instance._z_cols = instance._tableau[:, n:-1]
        instance._phase_col = instance._tableau[:, -1]
        instance._destabilizer_rows = instance._tableau_without_phase[
            : instance.isotropic_dim
        ]
        instance._stabilizer_rows = instance._tableau_without_phase[
            instance.isotropic_dim: 2 * instance.isotropic_dim
        ]
        instance._jw_elements_rows = instance._tableau_without_phase[
            2 * instance.isotropic_dim:
        ]
        return instance

    @classmethod
    def is_cnc(cls, n: int, m: int, tableau: np.ndarray) -> bool:
        # Check if the tableau has the correct shape
        if tableau.shape != (2 * n + 1, 2 * n + 1):
            logging.info("Tableau has incorrect shape.")
            return False

        # Check tableau only contains 0s and 1s
        if not np.all(np.isin(tableau, [0, 1])):
            logging.info("Tableau contains elements other than 0 and 1.")
            return False

        isotropic_dim = n - m
        tableau_without_phase = tableau[:, :-1]

        # Check commutation relations
        commutation_matrix = (
            tableau_without_phase @ symplectic_matrix(n) @
            tableau_without_phase.T
        ) % 2

        destabilizer_destabilizer_commutations = commutation_matrix[
            : isotropic_dim, : isotropic_dim
        ]
        if not np.array_equal(
            destabilizer_destabilizer_commutations,
            np.zeros((isotropic_dim, isotropic_dim))
        ):
            logging.info("Destabilizers do not commute with each other.")
            return False

        destabilizer_stabilizer_commutations = commutation_matrix[
            : isotropic_dim, isotropic_dim: 2 * isotropic_dim
        ]
        if not np.array_equal(
            destabilizer_stabilizer_commutations,
            np.eye(isotropic_dim)
        ):
            logging.info(
                "Destabilizier and stabilizer bases are not symplectic bases."
                )
            return False

        destabilizer_jw_elements_commutations = commutation_matrix[
            : isotropic_dim, 2 * isotropic_dim:
        ]
        if not np.array_equal(
            destabilizer_jw_elements_commutations,
            np.zeros((isotropic_dim, 2*m + 1))
        ):
            logging.info("Destabilizers do not commute with JW elements.")
            return False

        stabilizer_stabilizer_commutations = commutation_matrix[
            isotropic_dim: 2 * isotropic_dim, isotropic_dim: 2 * isotropic_dim
        ]
        if not np.array_equal(
            stabilizer_stabilizer_commutations,
            np.zeros((isotropic_dim, isotropic_dim))
        ):
            logging.info("Stabilizers do not commute with each other.")
            return False

        stabilizer_jw_elements_commutations = commutation_matrix[
            isotropic_dim: 2 * isotropic_dim, 2 * isotropic_dim:
        ]
        if not np.array_equal(
            stabilizer_jw_elements_commutations,
            np.zeros((isotropic_dim, 2*m + 1))
        ):
            logging.info("Stabilizers do not commute with JW elements.")
            return False

        jw_elements_jw_elements_commutations = commutation_matrix[
            2 * isotropic_dim:, 2 * isotropic_dim:
        ]
        expected = np.ones((2*m + 1, 2*m + 1), dtype=np.uint8)
        expected -= np.eye(2*m + 1, dtype=np.uint8)
        if not np.array_equal(jw_elements_jw_elements_commutations, expected):
            logging.info("JW elements do not anticommute with each other.")
            return False

        # Check independence
        destabilizer_rows = GF2(
            tableau_without_phase[: isotropic_dim]
            )
        if np.linalg.matrix_rank(destabilizer_rows) != isotropic_dim:
            logging.info("Destabilizer rows are not linearly independent.")
            return False

        stabilizer_rows = GF2(
            tableau_without_phase[isotropic_dim: 2*isotropic_dim]
            )
        if np.linalg.matrix_rank(stabilizer_rows) != isotropic_dim:
            logging.info("Stabilizer rows are not linearly independent.")
            return False
        jw_elements_rows = GF2(
            tableau_without_phase[2*isotropic_dim:]
            )
        if np.linalg.matrix_rank(jw_elements_rows) != 2*m:
            logging.info("JW elements rows do not have rank 2m.")
            return False

        all_rows = GF2(tableau_without_phase)
        if np.linalg.matrix_rank(all_rows) != 2*n:
            logging.info("All rows do not span the whole space.")
            return False

        return True

    @staticmethod
    def initial_cnc_tableau(n: int, m: int) -> np.ndarray:
        isotropic_dim = n - m
        row_count = 2 * n + 1
        column_count = 2 * n + 1

        tableau = np.zeros((row_count, column_count), dtype=np.uint8)

        # Define submatrices A and B
        A = np.tril(np.ones((m, m), dtype=np.uint8))
        B = np.tril(np.ones((m, m), dtype=np.uint8), -1)

        # Place the first identity matrix
        identity_pos1 = (slice(isotropic_dim), slice(n, n + isotropic_dim))
        tableau[identity_pos1] = np.eye(isotropic_dim, dtype=np.uint8)

        # Place the second identity matrix
        identity_pos2 = (
            slice(isotropic_dim, 2 * isotropic_dim),
            slice(isotropic_dim),
        )
        tableau[identity_pos2] = np.eye(isotropic_dim, dtype=np.uint8)

        # Place the A and B matrices
        A_pos1 = (
            slice(2 * isotropic_dim, 2 * isotropic_dim + m),
            slice(isotropic_dim, isotropic_dim + m),
        )
        tableau[A_pos1] = A
        A_pos2 = (
            slice(2 * isotropic_dim + m, 2 * isotropic_dim + 2 * m),
            slice(isotropic_dim + n, isotropic_dim + n + m),
        )
        tableau[A_pos2] = A
        B_pos1 = (
            slice(2 * isotropic_dim, 2 * isotropic_dim + m),
            slice(isotropic_dim + n, isotropic_dim + n + m),
        )
        tableau[B_pos1] = B
        B_pos2 = (
            slice(2 * isotropic_dim + m, 2 * isotropic_dim + 2 * m),
            slice(isotropic_dim, isotropic_dim + m),
        )
        tableau[B_pos2] = B

        # Place the last row
        last_row = np.zeros(column_count, dtype=np.uint8)
        last_row[n - m: n] = 1
        last_row[2 * n - m: 2 * n] = 1
        tableau[-1] = last_row

        return tableau

    @property
    def n(self) -> int:
        return self._n

    @property
    def m(self) -> int:
        return self._m

    @property
    def isotropic_dim(self) -> int:
        return self._isotropic_dim

    @property
    def tableau(self) -> np.ndarray:
        return self._tableau

    def apply_cnot(self, control_qubit: int, target_qubit: int) -> None:
        logging.debug(
            f"Applying CNOT with control_qubit={control_qubit}, "
            f"target_qubit={target_qubit}"
        )
        assert control_qubit != target_qubit

        self._phase_col ^= (
            self._x_cols[:, control_qubit]
            & self._z_cols[:, target_qubit]
            & (
                self._x_cols[:, target_qubit]
                ^ self._z_cols[:, control_qubit]
                ^ True
            )
        )
        self._x_cols[:, target_qubit] ^= self._x_cols[:, control_qubit]
        self._z_cols[:, control_qubit] ^= self._z_cols[:, target_qubit]

    def apply_hadamard(self, qubit: int) -> None:
        logging.debug(f"Applying Hadamard on qubit={qubit}")
        self._phase_col ^= self._x_cols[:, qubit] & self._z_cols[:, qubit]
        self._x_cols[:, qubit] ^= self._z_cols[:, qubit]
        self._z_cols[:, qubit] ^= self._x_cols[:, qubit]
        self._x_cols[:, qubit] ^= self._z_cols[:, qubit]

    def apply_phase(self, qubit: int) -> None:
        logging.debug(f"Applying Phase on qubit={qubit}")
        self._phase_col ^= self._x_cols[:, qubit] & self._z_cols[:, qubit]
        self._z_cols[:, qubit] ^= self._x_cols[:, qubit]

    def measure(self, measurement_basis: np.ndarray) -> int:
        logging.debug(f"Measuring with measurement_basis={measurement_basis}")
        assert len(measurement_basis) == 2 * self._n

        commutations_with_meas_basis = (
            self._tableau_without_phase
            @ (self._symplectic_matrix @ measurement_basis)
        ) % 2

        commutation_with_destabilizers = commutations_with_meas_basis[
            : self.isotropic_dim
        ]
        commutation_with_stabilizers = commutations_with_meas_basis[
            self.isotropic_dim: 2 * self.isotropic_dim
        ]
        commutation_with_jw_elements = commutations_with_meas_basis[
            2 * self.isotropic_dim:
        ]

        commutes_with_stabilizer = np.all(commutation_with_stabilizers == 0)
        commutes_with_jw_elements = np.all(commutation_with_jw_elements == 0)

        number_of_anticommuting_jw_elements = np.sum(
            commutation_with_jw_elements
        )

        outcome = 0
    
        # Case 1: Measurement basis is in the stabilizer
        if commutes_with_stabilizer and commutes_with_jw_elements:
            #print("Performing Case 1\n")
            logging.debug("Measurement basis is in the stabilizer.")
            temp_vec = np.zeros(2 * self._n, dtype=np.uint8)

            for i in range(self._isotropic_dim):
                if commutation_with_destabilizers[i] != 0:
                    e_i = self._stabilizer_rows[i]
                    # phase of the e_i in the stabilizer
                    s_i = self._phase_col[self.isotropic_dim + i]

                    outcome = outcome ^ s_i ^ beta(e_i, temp_vec)
                    temp_vec = temp_vec ^ e_i
            
            #if not self.is_cnc(self.n,self.m,self._tableau):
            #    print(f"Performing Case I. Update is incorrect.")

        # Case 2: Measurement basis is Omega but not in the stabilizer
        elif commutes_with_stabilizer and (
            number_of_anticommuting_jw_elements == 2 * self._m
        ):
            #print("Performing Case 2\n")
            logging.debug(
                "Measurement basis is in Omega but not in the stabilizer."
            )
            # Find the unique JW element that commutes with
            # the measurement basis
            indices = np.where(commutation_with_jw_elements == 0)[0]
            if indices.size > 0:
                k = indices[0]
            else:
                raise RuntimeError(
                    "No JW element commutes with the measurement basis."
                )
            a_k = self._jw_elements_rows[k]
            temp_vec = a_k
            # set outcome as the phase of the JW element a_k
            outcome = self._phase_col[2 * self._isotropic_dim + k]

            for i in range(self._isotropic_dim):
                if commutation_with_destabilizers[i] != 0:
                    e_i = self._stabilizer_rows[i]
                    # phase of the e_i in the stabilizer
                    s_i = self._phase_col[self.isotropic_dim + i]

                    outcome = outcome ^ s_i ^ beta(e_i, temp_vec)
                    temp_vec = temp_vec ^ e_i

            # With probability 1/2, flip the phase of the non-commuting
            # JW elements
            if self._rng.integers(2) == 1:
                jw_elements_start_index = 2 * self.isotropic_dim
                indices = np.arange(2 * self.m + 1)
                indices = indices[indices != k]
                self._phase_col[jw_elements_start_index + indices] ^= 1
            
            #if not self.is_cnc(self.n,self.m,self._tableau):
            #    print(f"Performing Case II. Update is incorrect.")

        # Case 3: Measurement basis commutes with the stabilizer
        # but not in Omega
        elif commutes_with_stabilizer:
            #print("Performing Case 3\n")
            logging.debug(
                "Measurement basis commutes with "
                "the stabilizer but not in Omega."
            )
            t = int(number_of_anticommuting_jw_elements // 2)

            logging.debug(
                f"Number of anticommuting JW elements:"
                f"{number_of_anticommuting_jw_elements}, so t={t}"
            )
            #print(f"t = {t}\n")
            # Reorganize rows of JW elements to have anticommuting JW elements
            # in the first 2t rows
            max_anticommuting_index = 2 * self.m
            min_commuting_index = 0
            while max_anticommuting_index >= 2 * t:
                # Find the next commuting element
                while min_commuting_index < 2 * self.m and (
                    commutation_with_jw_elements[min_commuting_index] == 1
                ):
                    min_commuting_index += 1

                # Find the next anticommuting element
                while max_anticommuting_index >= 2 * t and (
                    commutation_with_jw_elements[max_anticommuting_index] == 0
                ):
                    max_anticommuting_index -= 1

                if min_commuting_index < 2 * t and (
                    max_anticommuting_index >= 2 * t
                ):
                    # Find their index in the tableau and swap rows
                    i = min_commuting_index + 2 * self.isotropic_dim
                    j = max_anticommuting_index + 2 * self.isotropic_dim

                    self._tableau[[i, j]] = self._tableau[[j, i]]
                    min_commuting_index += 1
                    max_anticommuting_index -= 1

            anticomm_jw_indexes = range(
                2 * self.isotropic_dim, 2 * self.isotropic_dim + 2 * t
            )
            comm_jw_indexes = range(
                2 * self.isotropic_dim + 2 * t, 2 * self.n + 1
            )

            # New stabilizer, bar_b:
            for i in range(1,2*t):
                self._row_add_without_phase(
                    anticomm_jw_indexes[0], anticomm_jw_indexes[i]
                )
            
            # Find outcome with coinflip and update the phase of b_bar
            outcome = self._rng.integers(0, 2, dtype=np.uint8)
            self._phase_col[anticomm_jw_indexes[0]] = outcome

            # Find new JW elements by adding b_bar into commuting JW elements
            for i in comm_jw_indexes:
                self._row_add_with_phase(i, anticomm_jw_indexes[0])

            # scratch space: O(n) memory:
            scratch = self._tableau_without_phase[anticomm_jw_indexes[0]]^self._tableau_without_phase[anticomm_jw_indexes[1]]

            # Find new stabilizers/destabilizers
            for i in range(1,t):
                # New stabilizer:
                self._tableau_without_phase[anticomm_jw_indexes[2*i]] ^= scratch
                # New destabilizer:
                self._tableau_without_phase[anticomm_jw_indexes[2*i+1]] ^= scratch

                if i < t-1:
                    scratch ^= self._tableau_without_phase[anticomm_jw_indexes[2*i]]
                    scratch ^= self._tableau_without_phase[anticomm_jw_indexes[2*i+1]]
                
                # assign phase to new stabilizers:
                phase_bit = self._rng.integers(0, 2, dtype=np.uint8)
                self._phase_col[anticomm_jw_indexes[2*i]] = phase_bit

                # assign zero to new destabilizers:
                self._phase_col[anticomm_jw_indexes[2*i+1]] = np.uint8(0)
            

            # Rearrange stabilizers and destabilizers
            previous_stabilizer_indexes = [
                i for i in range(self.isotropic_dim, 2 * self.isotropic_dim)
            ]

            new_destabilizer_indexes = [
                2*self._isotropic_dim+2*i+1
                for i in range(t
                )
            ]

            new_stabilizer_indexes = [
                2*self._isotropic_dim+2*i
                for i in range(t
                )
            ]

            target_destabilizer_indexes = [i for i in range(2*self._isotropic_dim, 2 * self.isotropic_dim + t)]
            target_stabilizer_indexes = [i for i in range(2*self._isotropic_dim+t, 2 * self.isotropic_dim + 2*t)]

            # put new stabilizers and destabilizers in consecutive indices:
            self._tableau[
                target_stabilizer_indexes + target_destabilizer_indexes
            ] = self._tableau[
                new_stabilizer_indexes + new_destabilizer_indexes
            ]
            
            # put new stabilizers/destabilizers into last t rows
            self._tableau[
                previous_stabilizer_indexes + target_destabilizer_indexes
            ] = self._tableau[
                target_destabilizer_indexes + previous_stabilizer_indexes
            ]

            # Update tableau variables and subtables
            self._m -= t
            self._isotropic_dim = self.n - self.m
            self._destabilizer_rows = self._tableau_without_phase[
                : self.isotropic_dim
            ]
            self._stabilizer_rows = self._tableau_without_phase[
                self.isotropic_dim: 2 * self.isotropic_dim
            ]
            self._jw_elements_rows = self._tableau_without_phase[
                2 * self.isotropic_dim:
            ]
            #if not self.is_cnc(self.n,self.m,self._tableau):
            #    print(f"Performing Case III. Update is incorrect.")

        # Case 4: Measurement basis does not commute with the stabilizer
        else:
            #print("Performing Case 4\n")
            logging.debug(
                "Measurement basis does not commute with the stabilizer."
            )
            # Find the element in the stabilizer with the least index that
            # anticommutes with the measurement basis
            indices = np.where(commutation_with_stabilizers == 1)[0]
            if indices.size > 0:
                k = indices[0]
            else:
                raise RuntimeError(
                    "No Stabilizer element anticommutes with the "
                    "measurement basis."
                )
            p = k + self.isotropic_dim

            # Add p to every other anticommuting row in the tableau
            for i in range(2 * self.n + 1):
                if commutations_with_meas_basis[i] == 1 and i != p:
                    # If it is in the destabilizer we do not care about
                    # the phase column
                    if i < self.isotropic_dim:
                        self._row_add_without_phase(i, p)

                    # Otherwise we care about the phase column
                    else:
                        self._row_add_with_phase(i, p)

            # Replace p-th destabilizer with p-th stabilizer.
            # We do not care about phases of the destabilizers
            self._tableau_without_phase[p - self.isotropic_dim] = (
                self._tableau_without_phase[p]
            )

            # Replace p-th stabilizer with the measurement basis
            self._tableau_without_phase[p] = measurement_basis

            # Find outcome with coinflip and update the phase of the p-th
            # stabilizer accordingly
            outcome = self._rng.integers(0, 2, dtype=np.uint8)
            self._phase_col[p] = outcome
            #if not self.is_cnc(self.n,self.m,self._tableau):
            #    print(f"Performing Case IV. Update is incorrect.")

        logging.debug(f"Measurement outcome: {outcome}")

        return outcome

    def _row_add_without_phase(self, i: int, j: int) -> None:
        """
        Adds row j to row i in the tableau but does not change the phase
        column.
        """
        self._tableau_without_phase[i] ^= self._tableau_without_phase[j]

    def _row_add_with_phase(self, i: int, j: int) -> None:
        """
        Adds row j to row i in the tableau and changes the phase column
        accordingly.
        """
        self._row_add_without_phase(i, j)
        a = self._tableau_without_phase[i]
        b = self._tableau_without_phase[j]
        s = self._phase_col[i]
        r = self._phase_col[j]
        self._phase_col[i] = s ^ r ^ beta(a, b)
    
