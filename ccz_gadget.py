"""
Created on Sun Dec  1 19:17:12 2024

@author: ASUS
"""

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit import Aer, execute

def custom_toffoli_decomposition(qc, c1, c2, anc1, anc2, target,ccz_count):
    """
    Decompose a Toffoli gate using a custom circuit structure.
    
    Parameters:
        qc (QuantumCircuit): The quantum circuit to append the decomposition to.
        c1 (int): Index of the first control qubit.
        c2 (int): Index of the second control qubit.
        anc1 (int): Index of the first ancilla qubit.
        anc2 (int): Index of the second ancilla qubit.
        target (int): Index of the target qubit.
    """
    # Begin custom decomposition
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
    qc.cx(anc1,c2)
    qc.cx(c1, anc2)
    
    qc.h(anc1)
    qc.s(anc1)
    qc.cx(anc1, target)
    qc.h(anc1)
    creg_toff = ClassicalRegister(1, 'creg_toff_'+str(ccz_count))
    qc.add_register(creg_toff)
    # Measure ancilla1 and store the result in the specified classical bit
    qc.measure(anc1, creg_toff)
    
    #Apply Z gates to c1 and c2 if both ancillas measured as 1
    qc.cz(c1,c2).c_if(qc.cregs[0], 1)
    qc.h(target)
    # Reset ancilla qubits to |0> if you plan to reuse them

from qiskit import QuantumRegister, ClassicalRegister

def replace_toffoli_with_custom_decomposition(original_qc):
    # Count the number of Toffoli gates in the original circuit
    num_ccx = sum(1 for gate in original_qc.data if gate[0].name == "ccz")

    # Create a new circuit with additional ancilla qubits (2 per Toffoli gate)
    new_num_qubits = original_qc.num_qubits + 2 * num_ccx
    new_num_clbits = original_qc.num_clbits
    new_qc = QuantumCircuit(new_num_qubits, new_num_clbits)

    # List to keep track of ancilla qubit indices
    ancilla_indices = list(range(original_qc.num_qubits, new_num_qubits))
    ancilla_ptr = 0  # Pointer to assign ancillas

    # Map original qubits and classical bits
    qubit_mapping = {q: i for i, q in enumerate(original_qc.qubits)}
    clbit_mapping = {c: i for i, c in enumerate(original_qc.clbits)}

    ccz_count = 0

    # Iterate through the original circuit's gates
    for gate in original_qc.data:
        instruction, qargs, cargs = gate

        if instruction.name == "ccz":
            ccz_count += 1; print(ccz_count)
            # Extract qubit indices for the Toffoli gate
            c1 = qubit_mapping[qargs[0]]
            c2 = qubit_mapping[qargs[1]]
            target = qubit_mapping[qargs[2]]

            # Assign two ancilla qubits for this Toffoli gate
            anc1 = ancilla_indices[ancilla_ptr]
            anc2 = ancilla_indices[ancilla_ptr + 1]
            ancilla_ptr += 2

            # Append the custom decomposition using the assigned ancillas
            custom_toffoli_decomposition(new_qc, c1, c2, anc1, anc2, target,ccz_count)
        else:
            # Map qubit indices to the new circuit
            mapped_qubits = [qubit_mapping[q] for q in qargs]
            mapped_clbits = [clbit_mapping[c] for c in cargs]
            # Append the original gate
            new_qc.append(instruction, mapped_qubits, mapped_clbits)

    return new_qc


    """
    Replace all Toffoli gates in the original circuit with the custom decomposition.

    Parameters:
        original_qc (QuantumCircuit): The original quantum circuit containing Toffoli gates.

    Returns:
        QuantumCircuit: A new quantum circuit with Toffoli gates replaced by the custom decomposition.
    """
    # Count the number of Toffoli gates in the original circuit
    num_ccx = sum(1 for gate in original_qc.data if gate[0].name == "ccz")
    
    # Create a new circuit with additional ancilla qubits (2 ancillas per Toffoli gate)
    new_qc = QuantumCircuit(original_qc.num_qubits + 2 * num_ccx, original_qc.num_clbits)
    
    # Mapping from original qubits to new qubits
    original_qubits = original_qc.qubits
    original_clbits = original_qc.clbits
    
    # Define qubit mapping: original qubits retain their indices
    for i, qubit in enumerate(original_qubits):
        new_qc.add_register(original_qc.qubits[i].register)

    # Add ancilla qubits
    ancilla_qubits = [Qubit() for _ in range(2 * num_ccx)]
    new_qc.add_qubits(ancilla_qubits)
    
    # Mapping for ancilla qubits
    ancilla_start_index = original_qc.num_qubits
    ancilla_indices = list(range(ancilla_start_index, ancilla_start_index + 2 * num_ccx))
    
    # Initialize a variable to track ancilla allocation
    ancilla_ptr = 0
    
    # Iterate through the original circuit's gates
    for gate in original_qc.data:
        if gate[0].name == "ccz":
            # Extract qubits involved in the Toffoli gate
            control_qubit1 = gate[1][0]
            control_qubit2 = gate[1][1]
            target_qubit = gate[1][2]
            
            # Get their indices in the original circuit
            c1 = original_qc.qubits.index(control_qubit1)
            c2 = original_qc.qubits.index(control_qubit2)
            target = original_qc.qubits.index(target_qubit)
            
            # Assign two ancilla qubits for this Toffoli gate
            anc1 = ancilla_indices[ancilla_ptr]
            anc2 = ancilla_indices[ancilla_ptr + 1]
            ancilla_ptr += 2
            
            # Append the custom decomposition using the assigned ancillas
            custom_toffoli_decomposition(new_qc, c1, c2, anc1, anc2, target)
        else:
            # For non-Toffoli gates, append them directly
            # Map qubits: get their indices in the new circuit
            qubits_indices = [original_qc.qubits.index(q) for q in gate[1]]
            clbits_indices = [original_qc.clbits.index(c) for c in gate[2]]
            new_qc.append(gate[0], qubits_indices, clbits_indices)
    
    # Handle measurements and other classical operations
    for i, clbit in enumerate(original_qc.clbits):
        for gate in original_qc.data:
            if gate[0].name in ["measure"] and gate[2][0] == clbit:
                # Find the qubit being measured
                measured_qubit = gate[1][0]
                qubit_index = original_qc.qubits.index(measured_qubit)
                new_qc.measure(qubit_index, i)
    
    return new_qc

if __name__ == "__main__":
    from qiskit import Aer, execute, QuantumCircuit, ClassicalRegister

    # Create a sample circuit
    original_circuit = QuantumCircuit(3, 3)  # 3 qubits, 3 classical bits
    original_circuit.x([0,1,2])
    original_circuit.ccz(0, 1, 2)
    original_circuit.ccz(0,1,2)

    print("Original Circuit:")
    print(original_circuit.draw())

    # Replace Toffoli gates with custom decomposition
    transformed_circuit = replace_toffoli_with_custom_decomposition(original_circuit)

    # Add measurements for qubits 0, 1, 2
    for qubit in range(3):
        transformed_circuit.measure(qubit, qubit)  # Measure qubit i into classical bit i

    print("\nTransformed Circuit:")
    print(transformed_circuit.draw())
    
    # Save the QASM representation to a file
    with open("transformed_circuit.qasm", "w") as file:
        file.write(transformed_circuit.qasm())

    print("Transformed circuit QASM saved to 'transformed_circuit.qasm'")

    # Simulate the circuit
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(transformed_circuit, simulator, shots=1024)
    result = job.result()
    counts = result.get_counts()

    print("\nSimulation Results:")
    print(counts)
