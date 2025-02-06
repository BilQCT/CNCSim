import numpy as np
import h5py
from src import tableau_helper_functions as helper
from src import utils
from src import updated_cnc_tableau as cnc
import contextlib

# Current directory
#current_dir = os.getcwd()
# One upper directory
#upper_dir = os.path.dirname(current_dir)

# Redirect printed output to a file
with open("generating_stab_tableaus.txt", "w") as f_out:
    with contextlib.redirect_stdout(f_out):
        # set n <= K
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

        for n in range(1,K+1):
            print(f"Processing Stabilizer vectors for n = {n}:\n")

            # generate symplectic form for n -qubits:
            symplectic_form = helper.create_symplectic_form(n)

            with h5py.File(filename, "r") as f:
                key = f"n={n}"
                data = np.array(list(f[key]))

            N = data.shape[1]
            W = np.array(data[0,:])
            M = np.array(data[1:,:],dtype=np.int64)

            # quantum state from decomposition:
            rho_decomposition = sum(data[0,i]*data[1:,i] for i in range(data.shape[1]))

            # theoretical quantum state:
            rho_t = np.array([1,1/np.sqrt(2),1/np.sqrt(2),0])
            rho_theoretical = np.array([1.0])
            for _ in range(n):
                rho_theoretical = np.kron(rho_theoretical,rho_t)
            
            # round to same digit:
            rho_decomposition = np.round(rho_decomposition,8)
            rho_theoretical = np.round(rho_theoretical,8)

            #print("Quantum state from decomposition: \n")
            #print(rho_decomposition,"\n")
            #print("Quantum state from theory: \n")
            #print(rho_theoretical,"\n")

            print(f"Quantum state from decomposition agrees with theory? {np.all(rho_decomposition == rho_theoretical)} \n")

            #tableau_array = []
            results = []

            for i in range(N):
                print(f"Stabilizer in decomposition: {i+1}\n")
                print(f"Quasiprobability weight: {W[i]}\n")
                # Identify stabilizer elements and value assignment:
                omega = [(utils.Pauli.from_basis_order(n, j).bsf) for j in range((M.shape)[0]) if M[j,i] != 0]
                gamma = [to_bits[np.sign(M[j,i])] for j in range((M.shape)[0]) if M[j,i] != 0]

                # print omega elements:
                for a in omega:
                    print(f"Pauli Operator: {utils.pauli_bsf_to_str(a)}")

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
                # append zero row for empty JW elements:
                tableau = np.concatenate((tableau,np.zeros((1,tableau.shape[1]),dtype = np.int64)), axis = 0)

                # check if tableau is valid:
                cnc_sim = cnc.CncSimulator.from_tableau(n,0,tableau)
                print(f"\n Cnc tableau object: \n {cnc_sim}\n")
                #print(f"Cnc tableau:\n {tableau}\n")

                # map integer index to quasiprobability-tableau pair
                results.append((W[i], tableau, n, 0))

            # Collect results per tableau
            #results = []
            #for i in range(N):
            #    results.append((W[i], tableau_array[i], n, 0))

            # Convert to structured array with variable types
            combined = np.array(
                results, 
                dtype=[('W', float), ('tableau', 'O'), ('n', int), ('m', int)]
            )

            # Save the structured array
            print(f"Saving Stabilizer Tableau Keys: n = {n}: \n")
            np.save(f"./keys/stab_tableau_keys_{n}.npy", combined, allow_pickle=True)

            #print(f"Loading Stabilizer Tableau Keys: n = {n}: \n")
            #print(np.load(f"./keys/stab_tableau_keys_{n}.npy", allow_pickle=True))


