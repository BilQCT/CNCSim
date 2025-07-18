import numpy as np
import h5py
from src import tableau_helper_functions as helper
from src.cnc_simulator import CncSimulator as cnc

# set n <= K
K = 4
filename = "./keys/all_cnc_keys_4.h5"

# dictionary for mapping Pauli coefficients to bits:
to_bits = dict(zip([1, -1], [0, 1]))

for n in range(2, K+1):
    print(f"Processing CNC vectors for n = {n}:\n")

    # generate symplectic form for n -qubits:
    symplectic_form = helper.symplectic_matrix(n)

    with h5py.File(filename, "r") as f:
        key = f"n={n}"
        data = np.array(list(f[key]))

    N = data.shape[1]
    W = np.array(data[0, :])
    M = np.array(data[1:, :], dtype=np.int64)

    tableau_array = []
    m_array = []

    for i in range(N):
        # extract omega set and gamma images:
        omega = [
            (helper.get_pauli_vec_from_index(n, j).bsf)
            for j in range((M.shape)[0])
            if M[j, i] != 0
        ]
        gamma = [
            to_bits[np.sign(M[j, i])]
            for j in range((M.shape)[0])
            if M[j, i] != 0
        ]

        # Convert omega elements to tuples of int
        value_assignment = dict(
            (tuple(map(int, x)), gamma) for x, gamma in zip(omega, gamma)
        )

        # find m parameter from |Omega|:
        m = helper.find_m_from_omega_size(n, len(omega))
        m_array.append(m)

        # Determine the central elements and jw elements:
        stabilizer_set, non_stabilizer_set = helper.find_commuting_elements(
            omega)

        # Identify linearly independent vectors:
        stabilizer_gens = helper.find_independent_subset(stabilizer_set)

        jw_elements = helper.find_jw_elements(non_stabilizer_set)

        jw_basis = jw_elements[1:]

        # Generators of I_perp:
        normalizer_gens = np.concatenate((stabilizer_gens, jw_basis), axis=0)

        # Find vectors linearly indpendent from I_perp:
        complement_vectors = helper.find_complementary_subspace(
            normalizer_gens, n)

        # Modify those vectors to commpute with W:
        destabilizer_gens = helper.generate_destabilizer_basis(
            complement_vectors, jw_basis)

        # By definition a symplectic basis exists from Stab \oplus Destab:
        stab, destab = helper.symplectic_gram_schmidt(
            stabilizer_set,
            helper.generate_subspace_efficient(destabilizer_gens),
            n - m
        )

        # Check that the vector space is symplectic:
        symplectic_boolean = helper.is_symplectic(
            np.array(stab), np.array(destab), symplectic_form
        )

        # exit if not symplectic
        if not symplectic_boolean:
            raise ValueError(
                "Stabilizer and destabilizer must form a symplectic basis."
            )

        # Initialize tableau:
        tableau = np.concatenate((destab, stab, jw_elements), axis=0)

        # Extract generator phases
        phases = np.array(
            [
                value_assignment.get(tuple(map(int, tableau[i, :])), None)
                for i in range(tableau.shape[0])
            ]
        ).reshape(tableau.shape[0], 1)

        # Set destabilizer phases to 0
        phases[:len(destab), :] = np.zeros((len(destab), 1), dtype=np.int64)

        # Append phases to tableau
        tableau = np.concatenate((tableau, phases), axis=1)

        # check if proper cnc:
        if not cnc.is_cnc(n, m, tableau):
            raise ValueError("Not a proper CNC tableau.")

        # map integer index to quasiprobability-tableau pair
        tableau_array.append(tableau)

    # Collect results per tableau
    results = []
    for i in range(N):
        results.append((W[i], tableau_array[i], n, m_array[i]))

    # Convert to structured array with variable types
    combined = np.array(
        results,
        dtype=[('W', float), ('tableau', 'O'), ('n', int), ('m', int)]
    )

    print("Saving CNC tableaus.\n")
    # Save the structured array
    np.save(f"./deneme/cnc_tableau_keys_{n}.npy", combined, allow_pickle=True)

print("Generating CNC tableaus complete.\n")
