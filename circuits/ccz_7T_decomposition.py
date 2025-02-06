from qiskit import QuantumCircuit, Aer, execute, ClassicalRegister

# suppress warnings from qiskit 1.0
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Example code that raises warnings
warnings.warn("This is a warning!")

def ccz_7t_decomposition(qc, c1, c2, target):
    # layer 1:
    #qc.h(target)
    # layer 2
    qc.cnot(c2,target)
    # layer 3: tdg = sdg * t = s*s*s*t
    qc.s(target)
    qc.s(target)
    qc.s(target)
    qc.t(target)
    # layer 4
    qc.cnot(c1,target)
    # layer 5
    qc.t(target)
    # layer 6
    qc.cnot(c2,target)
    # layer 7: tdg
    qc.s(target)
    qc.s(target)
    qc.s(target)
    qc.t(target)
    # layer 8
    qc.cnot(c1,target)
    # layer 9
    qc.s(c2)
    qc.s(c2)
    qc.s(c2)
    qc.t(c2)
    qc.t(target)
    #qc.h(target)
    # layer 10
    qc.cnot(c1,c2)
    # layer 11: tdg
    qc.s(c2)
    qc.s(c2)
    qc.s(c2)
    qc.t(c2)
    # layer 12
    qc.cnot(c1,c2)
    # layer 13
    qc.t(c1)
    qc.s(c2)
    #qc.h(target)

def apply_ccz_via_7t_decomposition(original_qc):
    new_qc = QuantumCircuit(original_qc.num_qubits, original_qc.num_clbits)

    qubit_mapping = {q: i for i, q in enumerate(original_qc.qubits)}
    clbit_mapping = {c: i for i, c in enumerate(original_qc.clbits)}

    for gate in original_qc.data:
        instruction, qargs, cargs = gate

        if instruction.name == "ccz":
            c1 = qubit_mapping[qargs[0]]
            c2 = qubit_mapping[qargs[1]]
            target = qubit_mapping[qargs[2]]

            # Append custom decomposition
            ccz_7t_decomposition(
                new_qc, c1, c2,target)
        else:
            mapped_qubits = [qubit_mapping[q] for q in qargs]
            mapped_clbits = [clbit_mapping[c] for c in cargs]
            new_qc.append(instruction, mapped_qubits, mapped_clbits)

    return new_qc



if __name__ == "__main__":
    from qiskit import Aer, execute, QuantumCircuit, ClassicalRegister
    import numpy as np

    
    # Simple circuit to test state injection:
    n=3
    qc = QuantumCircuit(n,n)
    #qc.x(range(n))
    qc.ccz(0,1,2)
    #qc.ccz(0+m,1+m,2+m)

    print("Original Circuit:")
    print(qc.draw())

    # inject ccz state:
    qc_with_tgates = apply_ccz_via_7t_decomposition(qc)

    print("\State Injected Quantum Circuit with Measurements:")
    print(qc_with_tgates.draw())

    # Add measurements only for the original qubits (first 6)
    for qubit in range(n):  
        qc_with_tgates.measure(qubit, qubit)  # Measure each original qubit into its corresponding classical bit

    # Simulate the circuit
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc_with_tgates, simulator, shots=1024)
    result = job.result()
    raw_counts = result.get_counts()
    print("\nSimulation Results:")
    print(raw_counts)






def custom_ccz_decomposition(qc, c1, c2, anc1, anc2, target, clbit_index):
    qc.h(target)
    qc.h(anc1)
    qc.cx(c1, anc2)
    qc.cx(anc1, c1)
    qc.cx(anc1, c2)
    qc.cx(c2, anc2)
    qc.t(anc1)
    qc.t(anc2)
    
    qc.tdg(c2)
    qc.tdg(c1)
    qc.cx(c2, anc2)
    qc.cx(anc1, c1)
    qc.cx(anc1, c2)
    qc.cx(c1, anc2)
    
    qc.h(anc1)
    qc.s(anc1)
    qc.cx(anc1, target)
    qc.h(anc1)
    
    # Measure ancilla to the specified classical bit
    qc.measure(anc1, clbit_index)

    qc.h(c2)
    qc.cx(c1, c2).c_if(clbit_index, 1)
    qc.h(c2)
    qc.h(target)