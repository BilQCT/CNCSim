import numpy as np
import h5py
from src import tableau_helper_functions as helper
from src import utils

# Current directory
#current_dir = os.getcwd()
# One upper directory
#upper_dir = os.path.dirname(current_dir)

# set n <= K
K = 4
filename = "./keys/all_stab_keys_4.h5"
#filename = os.path.join(upper_dir,data_file)

'''
# Current directory
current_dir = os.getcwd()
# One upper directory
upper_dir = os.path.dirname(current_dir)
# Define the path to the target folder
main_dir = os.path.abspath(upper_dir)
# Define the path to the target folder
src_dir = os.path.abspath(os.path.join(os.getcwd(), "src"))
# Add the target folder to sys.path
sys.path.append(src_dir)
sys.path.append(main_dir)

from src import tableau_helper_functions as helper
from src import utils
import neater_cnc_tableau as cnc
'''

# set n <= K
K = 4
#data_file = "/keys/all_stab_keys_4.h5"
#filename = os.path.join(main_dir,data_file)
#print(filename)

# dictionary for mapping Pauli coefficients to bits:
to_bits = dict(zip([1,-1],[0,1]))

for n in range(2,K+1):
    print(f"Processing Stabilizer vectors for n = {n}:\n")

    # generate symplectic form for n -qubits:
    symplectic_form = helper.create_symplectic_form(n)

    with h5py.File(filename, "r") as f:
        key = f"n={n}"
        data = np.array(list(f[key]))

    N = data.shape[1]
    W = np.array(data[0,:])
    M = np.array(data[1:,:],dtype=np.int64)

    tableau_array = []

    for i in range(N):
        # Identify stabilizer elements and value assignment:
        omega = [(utils.Pauli.from_basis_order(n, j).bsf) for j in range((M.shape)[0]) if M[j,i] != 0]
        gamma = [to_bits[np.sign(M[j,i])] for j in range((M.shape)[0]) if M[j,i] != 0]

        # Convert omega elements to tuples of int
        value_assignment = dict(
        (tuple(map(int, x)), gamma) for x, gamma in zip(omega, gamma)
        )

        # For stabilizer m=0:
        m = 0

        # determine the central elements and jw elements:
        stabilizer_set = helper.find_commuting_elements(omega)
        
        # Identify linearly independent vectors:
        stabilizer_gens = helper.find_independent_subset(stabilizer_set)

        # Find vectors linearly independent from stab:
        complement_vectors = helper.find_complementary_subspace(stabilizer_gens,n)
        # Construct symplectic basis
        stab, destab = helper.symplectic_gram_schmidt(stabilizer_set,complement_vectors,n-m)
        # Check that basis is symplectic
        symplectic_boolean = helper.is_symplectic(np.array(stab),np.array(destab),symplectic_form)
        
        # Create tableau:
        tableau = np.concatenate((destab,stab),axis=0)
        
        # Extract generator phases
        phases = np.array(
        [value_assignment.get(tuple(map(int, tableau[i, :])), None) for i in range(tableau.shape[0])]
        ).reshape(tableau.shape[0], 1)
        # Set destabilizer phases to zero:
        phases[:len(destab),:] = np.zeros((len(destab),1),dtype = int)
        # Append phases to tableau:
        tableau = np.concatenate((tableau,phases),axis = 1)

        # map integer index to quasiprobability-tableau pair
        tableau_array.append(tableau)

    # Collect results per tableau
    results = []
    for i in range(N):
        results.append((W[i], tableau_array[i], n, 0))

    # Convert to structured array with variable types
    combined = np.array(
        results, 
        dtype=[('W', float), ('tableau', 'O'), ('n', int), ('m', int)]
    )

    # Save the structured array
    np.save(f"./keys/stab_tableau_keys_{n}.npy", combined, allow_pickle=True)


