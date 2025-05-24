# hidden_shift_sweep.py
# Runs CNC-based hidden shift simulations over a grid of parameters (n, g, epsilon, prob_fail)
# using run_qcm, and records metrics: number of samples, negativity, born-rule estimates, and runtime.

import time
import csv
import itertools

# import the run_qcm function and any key-negativity utilities
from src.qcm_sim import run_qcm  # adjust import path as needed
from src.compile_keys import compute_total_negativity

# import circuits:
import circuits.hidden_shift as hshift 


def run_experiment(n, g, epsilon, prob_fail):
    """
    Execute the adaptive Clifford simulation via run_qcm, returning:
      - N: number of samples
      - negativity
      - born_rule_estimates
      - total_runtime
    """
    # Generate a random hidden shift
    shift = [1 for _ in range(n)]

    # create qiskit circuit
    qc = hshift.hidden_shift(n,g,shift,is_clifford = False)
    qc_qasm = qc.qasm()

    # measure end-to-end runtime
    start_all = time.perf_counter()
    # call run_qcm: assume file_loc etc. are configured inside run_qcm or use default MSI string
    outputs, born_rule_estimates, shot_times = run_qcm(
        msi_qasm_string=qc_qasm,
        hoeffding=True,
        epsilon=epsilon,
        prob_fail=prob_fail,
        shots=None  # ignored when hoeffding=True
    )
    total_runtime = time.perf_counter() - start_all

    N = len(outputs)
    # negativity can be recomputed or extracted:
    # assuming compile_keys.compute_total_negativity is consistent with run_qcm
    negativity = compute_total_negativity(g)

    return N, negativity, born_rule_estimates, total_runtime


def main():
    import time
    import csv
    import itertools

    # import the run_qcm function and any key-negativity utilities
    from src.qcm_sim import run_qcm  # adjust import path as needed
    from src.compile_keys import compute_total_negativity

    # import circuits:
    import circuits.hidden_shift as hshift


    # Define parameter grid
    Gs = [1, 2, 3, 4]                    # CCZ gate counts 3*2*
    Ns = [6, 106, 12, 112, 18, 118, 24, 124]                # total qubits (even)
    epsilons = [0.01, 0.1]           # Hoeffding epsilon
    prob_fails = [0.1, 0.2]           # Hoeffding failure thresholds

    outfile = 'hidden_shift_cnc_sweep.csv'
    # Open CSV and write header
    with open(outfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'n', 'g', 'epsilon', 'prob_fail',
            'N_samples', 'negativity', 'born_estimates', 'runtime_s'
        ])
        print(f"Logging results to {outfile}\n", flush=True)

        for n, g, eps, pf in itertools.product(Ns, Gs, epsilons, prob_fails):
            if n % 2 != 0 or 3*g > n//2:
                continue
            print(f"Running n={n}, g={g}, eps={eps}, prob_fail={pf}...", end=' ', flush=True)
            try:
                N, neg, born_est, rt = run_experiment(n, g, eps, pf)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                continue

            writer.writerow([n, g, eps, pf, N, neg, born_est, rt])
            csvfile.flush()
            print("DONE", flush=True)

    print(f"\nAll done. Final results in {outfile}")

if __name__ == '__main__':
    main()
