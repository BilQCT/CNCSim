import numpy as np
import h5py
from src import tableau_helper_functions as helper
from src import utils
import os
import pickle

# Current directory
#current_dir = os.getcwd()
# One upper directory
#upper_dir = os.path.dirname(current_dir)
#print(upper_dir)


class Error(Exception):
    """Base class for other exceptions"""
    pass


class KeyNumberError(Error):
    """Exception raised when trying to generate keys that have already been generated.
    """
    def __init__(
        self,
        message="Keys already exist: Do not need to re-generate keys!"
    ):
        self.message = message
        super().__init__(self.message)


def get_current_key_status(n_target):
    
    # Current directory
    current_dir = os.getcwd()

    # key path:
    key_path = "keys/keys_info.pkl"

    # check current tableaus:
    file_path = os.path.join(current_dir,key_path)

    # To load it back
    with open(file_path, 'rb') as file:
        key_dict = pickle.load(file)
    
    # load existing key meta data:
    n_cnc = key_dict['cnc']
    n_stab = key_dict['stb']
    n_max = key_dict['max']

    # if existing: return True:
    if n_target <= n_max:
        #print("Keys already exist.\n")
        return True, n_target
    # if not existing: return False, determine which existing keys to compose with and how many times:
    else:
        # n_base = n_cnc+m*n_stab:
        m_stab = int(np.floor((n_max-n_cnc)/n_stab))
        n_base = n_cnc+m_stab*n_stab

        # Number of times to compose with biggest stabilizer decomposition:
        # n_target = n_base + r*n_stab + s
        r_stab = int(np.floor(n_target-n_base)/n_stab)
        s_stab = (n_target - n_base) % n_stab

        #print("New keys need to be generated for this problem.\n")
        return False, n_base, n_stab, r_stab, s_stab


def compile_and_save_new_keys(n_base, n_stab, r_stab, s_stab):

    # Path to keys:
    current_dir = os.getcwd()
    key_dir = os.path.join(current_dir, "keys")

    # Load n_base keys:
    #print(f"Loading keys for {n_base} qubits.\n")
    keys_n_base_path = os.path.join(key_dir, f"keys_n_{n_base}.npy")
    keys_n_base = np.load(keys_n_base_path, allow_pickle=True)

    # If r_stab = s_stab = 0: return n_base:
    if (r_stab == 0) and (s_stab == 0):
        return keys_n_base

    # Load n_stab keys:
    keys_n_stab_path = os.path.join(key_dir, f"stab_tableau_keys_{n_stab}.npy")
    keys_n_stab = np.load(keys_n_stab_path, allow_pickle=True)

    # Calculate target qubit count:
    n_target = n_base + r_stab * n_stab + s_stab
    #print(f"Generating new keys for {n_target} qubits.\n")

    # Initialize new tableaux:
    new_tableaus = list(keys_n_base)

    # Step 1: Apply r_stab repetitions with n_stab tableaus
    for k in range(r_stab):
        updated_tableaus = []
        for base_tableau in new_tableaus:
            n_cnc, m_cnc = base_tableau[2], base_tableau[3]
            for stab_tableau in keys_n_stab:
                # Calculate new quasiprobability:
                new_q = base_tableau[0] * stab_tableau[0]

                # Compose tableaus:
                composed_tableau = helper.compose_tableaus(
                    base_tableau[1], stab_tableau[1][:-1,:], m_cnc, 0
                )

                # Append the result:
                updated_tableaus.append((new_q, composed_tableau, n_cnc + n_stab, m_cnc))
        new_tableaus = updated_tableaus

        # Convert to numpy array with structured dtype
        new_tableau_array = np.array(
            new_tableaus,
            dtype=[('W', float), ('tableau', 'O'), ('n', int), ('m', int)]
        )

        # size of current tableaus:
        n_current = n_base+(k+1)*n_stab

        # Save the structured array
        #print(f"Saving keys for {n_current} qubits to file.\n")
        new_key_path = os.path.join(key_dir,f"keys_n_{n_current}.npy")
        np.save(new_key_path, new_tableau_array, allow_pickle=True)

        # exist loop if s_stab = 0:
        if s_stab == 0:
            return new_tableau_array

    # Step 2: Apply composition with s_stab tableaus
    if s_stab > 0:
        # load remaining stabilizer tail:
        keys_s_stab_path = os.path.join(key_dir, f"stab_tableau_keys_{s_stab}.npy")
        keys_s_stab = np.load(keys_s_stab_path, allow_pickle=True)

        final_tableaus = []
        for base_tableau in new_tableaus:
            n_cnc, m_cnc = base_tableau[2], base_tableau[3]
            for stab_tableau in keys_s_stab:
                # Calculate new quasiprobability:
                new_q = base_tableau[0] * stab_tableau[0]

                # Compose tableaus:
                composed_tableau = helper.compose_tableaus(
                    base_tableau[1], stab_tableau[1][:-1,:], m_cnc, 0
                )

                # Append the result:
                final_tableaus.append((new_q, composed_tableau, n_cnc + s_stab, m_cnc))
        new_tableaus = final_tableaus

    # Convert to numpy array with structured dtype
    new_tableau_array = np.array(
        new_tableaus,
        dtype=[('W', float), ('tableau', 'O'), ('n', int), ('m', int)]
    )

    # Save the structured array
    print(f"Saving keys for {n_target} qubits to file.\n")
    new_key_path = os.path.join(key_dir,f"keys_n_{n_target}.npy")
    np.save(new_key_path, new_tableau_array, allow_pickle=True)

    return new_tableau_array




def get_weighting_for_keys(keys):
    # retreive quasiprobabilities:
    quasiprobabilities = [keys[i][0] for i in range(keys.shape[0])]
    # compute negativity:
    negativity = sum(np.abs(x) for x in quasiprobabilities)
    # renormalized probability distribution:
    probabilities = [np.abs(q)/negativity for q in quasiprobabilities]
    # check if normalized:
    if np.round(sum(p for p in probabilities)) != 1.00:
        raise ValueError("Probabilities should be normalized")
    else:
        # return normalized probability distribution:
        return quasiprobabilities, probabilities




def get_keys(n_target):
    #print(f"Retrieving {n_target} qubit keys.\n")

    # retreive status of n_target in database:
    key_tuple = get_current_key_status(n_target)

    # check if n_target in database:
    if key_tuple[0]:
        #print("Keys are present, loading keys.\n")
        keys_n_target = np.load(f"./keys/keys_n_{n_target}.npy",allow_pickle = True)
    
    # if not:
    else:
        # generate keys from existing keys via composition:
        #print("Keys are not present, generating keys.\n")
        keys_n_target = compile_and_save_new_keys(key_tuple[1], key_tuple[2], key_tuple[3], key_tuple[4])
        #print("Keys generated.\n")
    
    # extract probabilities:
    keys_quasiprobabilities, keys_probabilities = get_weighting_for_keys(keys_n_target)

    return keys_quasiprobabilities, keys_probabilities, keys_n_target




# Example usage
if __name__ == "__main__":
    for t in range(5,11):
        get_keys(t)