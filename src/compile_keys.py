import numpy as np
from src import tableau_helper_functions as helper
import os
import random

#################################################################
#                                                               #
# Compute probabilities from quasi-distribution                 #
#                                                               #                                                            
#################################################################

def get_key_probabilities(quasiprobabilities):
    """
    Compute and return a normalized probability distribution from a list of quasi-probabilities.
    
    The function first computes the negativity, which is defined as the sum of the absolute 
    values of all provided quasi-probabilities. Then, each probability is computed as the absolute 
    value of the corresponding quasi-probability divided by the negativity. If the sum of the 
    computed probabilities is not (approximately) 1, a ValueError is raised.
    
    Parameters
    ----------
    quasiprobabilities : list or array_like of float
        A collection of quasi-probability weights (which may be negative).
    
    Returns
    -------
    probabilities : list of float
        The normalized probability distribution such that the sum of all probabilities is 1.
    
    Raises
    ------
    ValueError
        If the computed probabilities do not sum to 1 (within rounding tolerance).
    """
    # compute negativity:
    negativity = sum(np.abs(x) for x in quasiprobabilities)
    # renormalized probability distribution:
    probabilities = [np.abs(q) / negativity for q in quasiprobabilities]
    # check if normalized:
    if np.round(sum(p for p in probabilities)) != 1.00:
        raise ValueError("Probabilities should be normalized")
    else:
        return probabilities

#################################################################
#                                                               #
# Load keys for t-gates                                         #
#                                                               #                                                            
#################################################################

# Define number of keys to load for CNC and stabilizer tableaus.
n_cnc = 4
n_stab = 4

# Get the parent directory of the current directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname('src/compile_keys.py'), '..'))

# Construct the path to the "keys" directory located in the parent directory
key_dir = os.path.join(parent_dir, "keys")

# Initialize lists for CNC keys.
keys_cnc = []
probabilities_cnc = []
m_values_cnc = []
negativity_cnc = []

for n in range(1, n_cnc + 1):
    # Load CNC keys for a given n.
    keys_path = os.path.join(key_dir, f"keys_n_{n}.npy")
    keys = np.load(keys_path, allow_pickle=True)
    
    # Extract quasi-probabilities from each key.
    quasiprobabilities = [keys[i][0] for i in range(keys.shape[0])]
    # Compute normalized probabilities.
    probabilities = get_key_probabilities(quasiprobabilities)
    # Compute negativity.
    negativity = sum(np.abs(q) for q in quasiprobabilities)
    
    keys_cnc.append(keys)
    probabilities_cnc.append(probabilities)
    m_values_cnc.append([keys[i][-1] for i in range(keys.shape[0])])
    negativity_cnc.append(negativity)

# Initialize lists for stabilizer keys.
keys_stab = []
probabilities_stab = []
negativity_stab = []

for n in range(1, n_stab + 1):
    keys_path = os.path.join(key_dir, f"stab_tableau_keys_{n}.npy")
    keys = np.load(keys_path, allow_pickle=True)
    
    quasiprobabilities = [keys[i][0] for i in range(keys.shape[0])]
    probabilities = get_key_probabilities(quasiprobabilities)
    negativity = sum(np.abs(q) for q in quasiprobabilities)
    
    keys_stab.append(keys)
    probabilities_stab.append(probabilities)
    negativity_stab.append(negativity)

#################################################################
#                                                               #
# Compile keys using tableau composition                        #
#                                                               #                                                            
#################################################################

def get_key_breakdown(n_target):
    """
    Break down the target number of qubits into a key composition structure.
    
    The breakdown follows the formula:
    
        n_target = n_cnc + m_stab * n_stab + s_stab
    
    If n_target is less than or equal to n_cnc, the breakdown is defined as:
    
        (n_target, 0, 0)
    
    Otherwise, the function computes:
    
        m_stab = floor((n_target - n_cnc) / n_stab)
        n_base = n_cnc + m_stab * n_stab
        s_stab = (n_target - n_base) mod n_stab
    
    Parameters
    ----------
    n_target : int
        The target number of qubits for the composite key.
    
    Returns
    -------
    tuple of (int, int, int)
        A tuple (n_cnc, m_stab, s_stab) where:
            - n_cnc: The base number of qubits from the CNC key.
            - m_stab: The number of full stabilizer tableaus to compose.
            - s_stab: The remainder stabilizer qubits.
    """
    if n_target <= n_cnc:
        return n_target, 0, 0

    m_stab = int(np.floor((n_target - n_cnc) / n_stab))
    n_base = n_cnc + m_stab * n_stab
    s_stab = (n_target - n_base) % n_stab

    return n_cnc, m_stab, s_stab


def compute_total_negativity(n_target):
    """
    Compute the total negativity for a composite key based on the target qubit number.
    
    The total negativity is computed by breaking down n_target into:
        (k_cnc, r_stab, s_stab) = get_key_breakdown(n_target)
    
    Then, the negativity is computed as follows:
    
        - If r_stab == 0 and s_stab == 0:
              negativity = negativity_cnc[k_cnc - 1]
        - If r_stab == 0 and s_stab > 0:
              negativity = negativity_cnc[k_cnc - 1] * negativity_stab[s_stab - 1]
        - If r_stab > 0 and s_stab == 0:
              negativity = negativity_cnc[k_cnc - 1] * (negativity_stab[n_stab - 1] ** r_stab)
        - Otherwise:
              negativity = negativity_cnc[k_cnc - 1] * (negativity_stab[n_stab - 1] ** r_stab) * negativity_stab[s_stab - 1]
    
    Parameters
    ----------
    n_target : int
        The target number of qubits for the composite key.
    
    Returns
    -------
    float
        The total negativity computed for the composite key.
    """
    k_cnc, r_stab, s_stab = get_key_breakdown(n_target)
    
    negativity_k_cnc = negativity_cnc[k_cnc - 1]
    negativity_n_stab = negativity_stab[n_stab - 1]
    negativity_s_stab = negativity_stab[s_stab - 1]
    
    if (r_stab == 0) and (s_stab == 0):
        return negativity_k_cnc
    elif (r_stab == 0) and (s_stab > 0):
        return negativity_k_cnc * negativity_s_stab
    elif (r_stab > 0) and (s_stab == 0):
        return negativity_k_cnc * (negativity_n_stab ** r_stab)
    else:
        return negativity_k_cnc * (negativity_n_stab ** r_stab) * negativity_s_stab


def sample_single_key(n_target):
    """
    Sample and compose a single composite key using tableau composition.
    
    The function performs the following steps:
    
    1. Breaks down n_target into (k_cnc, r_stab, s_stab) via get_key_breakdown.
    2. Loads the CNC keys, and stabilizer keys for full stabilizer tableaus and the remainder, using the 
       pre-loaded global lists (keys_cnc and keys_stab).
    3. Samples one key index from the CNC keys based on their probability distribution.
    4. Initializes a composite tableau using the sampled CNC key.
    5. For each full stabilizer repetition (r_stab), samples a stabilizer tableau and composes it with the current tableau.
       The quasiprobability is updated multiplicatively.
    6. If there is a remainder (s_stab > 0), samples one additional stabilizer key and composes it.
    
    Parameters
    ----------
    n_target : int
        The target number of qubits for the composite key.
    
    Returns
    -------
    tuple
        A tuple (array, q, m) where:
            - array: The composed tableau (as a numpy array).
            - q: The total quasi-probability (product of the sampled keys' weights).
            - m: The m value associated with the CNC keys (from the sampled CNC key).
    """
    k_cnc, r_stab, s_stab = get_key_breakdown(n_target)
    
    # Load keys from global lists.
    keys_k_cnc = keys_cnc[k_cnc - 1]
    keys_n_stab = keys_stab[n_stab - 1]
    keys_s_stab = keys_stab[s_stab - 1]
    
    # Load probabilities from global lists.
    probabilities_k_cnc = probabilities_cnc[k_cnc - 1]
    probabilities_n_stab = probabilities_stab[n_stab - 1]
    probabilities_s_stab = probabilities_stab[s_stab - 1]
    
    # Create index lists.
    indices_k_cnc = list(range(len(keys_k_cnc)))
    indices_n_stab = list(range(len(keys_n_stab)))
    indices_s_stab = list(range(len(keys_s_stab)))
    
    # Sample one CNC key.
    sample_k_cnc = random.choices(indices_k_cnc, weights=probabilities_k_cnc, k=1)[0]
    
    # Initialize composite tableau from the sampled CNC key.
    array = keys_k_cnc[sample_k_cnc][1]
    q = keys_k_cnc[sample_k_cnc][0]
    m = m_values_cnc[k_cnc - 1][sample_k_cnc]
    
    # Step 1: Compose with r_stab stabilizer tableaus.
    for _ in range(r_stab):
        sample_n_stab = random.choices(indices_n_stab, weights=probabilities_n_stab, k=1)[0]
        stab_tableau = keys_n_stab[sample_n_stab][1][:-1, :]  # Exclude phase row if needed.
        array = helper.compose_tableaus(array, stab_tableau, m, 0)
        q = q * keys_n_stab[sample_n_stab][0]
    
    # Step 2: If there is a remainder, compose with one additional stabilizer tableau.
    if s_stab > 0:
        sample_s_stab = random.choices(indices_s_stab, weights=probabilities_s_stab, k=1)[0]
        stab_tableau = keys_s_stab[sample_s_stab][1][:-1, :]
        array = helper.compose_tableaus(array, stab_tableau, m, 0)
        q = q * keys_s_stab[sample_s_stab][0]
    
    return array, q, m
