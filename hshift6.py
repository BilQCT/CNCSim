
from qiskit import QuantumCircuit
from ccz_gadget import *
from qiskit import Aer, execute, QuantumCircuit, ClassicalRegister

def hshift6():

    z_list=[0,1,0,0,1,0] # list that is made to randomly apply Z gates to qubits in hidden shift algorithm
    qubits=list(range(6)) # list that is made to apply 6 hadamards easily
    cbits=qubits
    qc=QuantumCircuit(6,6)
    qc.h(qubits)
    qc.h(3)
    qc.cx(0,3)
    qc.h(3)
    qc.h(4)
    qc.cx(1,4)
    qc.h(4)
    qc.h(5)
    qc.cx(2,5) # For applying CZ, IH CX IH will be applied
    qc.h(5)
    qc.ccz(0, 1, 2)
    qc.h(qubits)
    qc.h(3)
    qc.cx(0,3)
    qc.h(3)
    qc.h(4)
    qc.cx(1,4)
    qc.h(4)
    qc.h(5)
    qc.cx(2,5)
    qc.h(5)
    qc.ccz(3, 4, 5)
    for i,z in enumerate(z_list):
        if z:
            qc.z(i)
    qc.h(qubits)
    qc.measure(qubits, cbits)
    print(qc)
    return qc

qc = hshift6()

# Simulate the circuit
simulator = Aer.get_backend('qasm_simulator')
job = execute(qc, simulator, shots=1024)
result = job.result()
counts = result.get_counts()

print("\nSimulation Results:")
print(counts)

qc_with_decomposition = replace_toffoli_with_custom_decomposition(qc)

print(qc_with_decomposition)

# Simulate the circuit
simulator = Aer.get_backend('qasm_simulator')
job = execute(qc_with_decomposition, simulator, shots=1024)
result = job.result()
counts = result.get_counts()

print("\nSimulation Results:")
print(counts)
