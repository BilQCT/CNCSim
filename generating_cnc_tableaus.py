import numpy as np
import h5py
import neater_cnc_tableau as cnc
from src import tableau_helper_functions as helper
from src import utils

# Current directory
#current_dir = os.getcwd()
# One upper directory
#upper_dir = os.path.dirname(current_dir)

# set n <= K
K = 4
filename = "./keys/all_cnc_keys_4.h5"
#filename = os.path.join(upper_dir,data_file)

# dictionary for mapping Pauli coefficients to bits:
to_bits = dict(zip([1,-1],[0,1]))

for n in range(2,K+1):
    print(f"Processing CNC vectors for n = {n}:\n")

    # generate symplectic form for n -qubits:
    symplectic_form = helper.create_symplectic_form(n)

    with h5py.File(filename, "r") as f:
        key = f"n={n}"
        data = np.array(list(f[key]))

    N = data.shape[1]
    W = np.array(data[0,:])
    M = np.array(data[1:,:],dtype=np.int64)

    tableau_array = []
    m_array = []

    for i in range(N):
        # extract omega set:
        omega = [(utils.Pauli.from_basis_order(n, j).bsf) for j in range((M.shape)[0]) if M[j,i] != 0]
        gamma = [to_bits[np.sign(M[j,i])] for j in range((M.shape)[0]) if M[j,i] != 0]
        #value_assignment = dict(zip([tuple(x) for x in omega],gamma))
        # Convert omega elements to tuples of int
        value_assignment = dict(
        (tuple(map(int, x)), gamma) for x, gamma in zip(omega, gamma)
        )

        # find m parameter from |Omega|:
        m = helper.find_m_from_omega(n,len(omega))
        m_array.append(m)

        # determine the central elements and jw elements:
        stabilizer_set, jw_set = helper.find_commuting_elements(omega)

        # identify linearly independent vectors:
        stabilizer_gens = helper.find_independent_subset(stabilizer_set)

        # find jw gens:
        cosets = helper.find_cosets(jw_set)
        jw_gens = [coset[0] for coset in cosets ]
        normalizer_gens = np.concatenate((stabilizer_gens,jw_gens[1:]),axis = 0)

        # find destabilizer:
        complement_vectors = helper.find_complementary_subspace(normalizer_gens,n)
        destabilizer_gens = helper.generate_destabilizer_basis(complement_vectors,jw_gens[1:])
        stab, destab = helper.symplectic_gram_schmidt(helper.generate_subspace_efficient(stabilizer_gens),helper.generate_subspace_efficient(destabilizer_gens),n-m)
        symplectic_boolean = helper.is_symplectic(np.array(stab),np.array(destab),symplectic_form)

        # create tableau:
        tableau = np.concatenate((destab,stab,jw_gens),axis=0)
        # extract generator phases
        phases = np.array(
        [value_assignment.get(tuple(map(int, tableau[i, :])), None) for i in range(tableau.shape[0])]
        ).reshape(tableau.shape[0], 1)
        # set destabilizer phases to 0
        phases[:len(destab),:] = np.zeros((len(destab),1),dtype = int)

        # full tableau
        tableau = np.concatenate((tableau,phases),axis = 1)

        # map integer index to quasiprobability-tableau pair
        tableau_array.append(tableau)

        # print results:
        print(f"m = {m}")
        print(f"Stabilizer generators: {[cnc.pauli_bsf_to_str(x) for x in (stab)]}")
        print(f"Jordan-Wigner elements: {[cnc.pauli_bsf_to_str(x) for x in jw_gens]}")
        print(f"Destabilizer generators: {[cnc.pauli_bsf_to_str(x) for x in (destab)]}")
        print(f"Symplectic pair: {(symplectic_boolean)} \n")

    # Collect results per tableau
    results = []
    for i in range(N):
        results.append((W[i], tableau_array[i], n, m_array[i]))

    # Convert to structured array with variable types
    combined = np.array(
        results, 
        dtype=[('W', float), ('tableau', 'O'), ('n', int), ('m', int)]
    )

    # Save the structured array
    np.save(f"./keys/cnc_tableau_keys_{n}.npy", combined, allow_pickle=True)


