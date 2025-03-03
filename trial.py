# suppress qiskit 1.0 deprecation warnings:
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Example code that raises warnings
warnings.warn("This is a warning!")

from qiskit import QuantumCircuit, Aer, execute
import qcm_sim_2 as sim
import qasm_prep as prep

# Create quantum circuit
qc = QuantumCircuit(1, 1)

n = 4

for _ in range(n):
    qc.t(0)
    qc.h(0)

# Measure all qubits
qc.measure(0,0)

qc_qasm = qc.qasm()

sim.run_qcm(qc_qasm)