import os
"""
--------------------------------------------------------------------------------
This module contains:

* Error classes: which characterize the different errors that can be raised if
  the input QASM quantum circuit (provided as a string) exhibits invalid properties.

* The QuCirc class: which describes a general QuCirc object characterized by the
  QASM string representing the quantum circuit. This means that an instance of this
  class (i.e. a specific quantum circuit) is defined by the provided QASM string.
  Once such an instance has been defined, the methods within the class can be applied to it.

The methods within this class are:

* Method 1: get_circuit_data(self)
            - This method processes the input QASM string provided by the user.
            - It formats the string by splitting it into lines and removing blank
              and commented lines.
            - It returns a list with all the relevant information in the QASM string,
              i.e., all the relevant information about the quantum circuit.

            - Summary: this method returns all the relevant information about the
              quantum circuit instance to which it is applied.

* Method 2: verify_input_circuit(self)
            - This method uses the information list produced by the previous method
              and verifies whether the input quantum circuit has a valid form. Namely,
              the circuit must:
                * have 1 qreg and 1 creg (a total of exactly 2 registers);
                * have at least 1 measurement;
                * be a Clifford+T circuit (only allowed operations are H, S, CNOT, T,
                  measure, and barrier);
                * have at least 1 T gate, otherwise it is a simple stabilizer circuit
                  (efficiently classically simulatable).
            - If the circuit is invalid, the method raises different errors depending
              on the condition that is not met.
            - If the circuit is valid, the method edits the list, changing the names of
              the quantum and classical registers to `q_stab` & `c_stab`, respectively.
              Then, the edited list, the number of qubits, the T-count, and the total
              number of measurements are returned.

            - Summary: this method either warns the user that the input quantum circuit
              instance is invalid or else (if the circuit is valid) it returns the properties
              of that instance.

* Method 3: msi_circuit(self, msi_clifford_file_loc, msi_clifford_file_name)
            - This method converts the Clifford+T circuit instance into a gadgetized
              (adaptive) Clifford circuit which is equivalent to the original one.
            - In doing so, it creates an auxiliary quantum register (`q_magic`) with
              as many qubits as the T-count and an auxiliary classical register (`c_magic`)
              for storing measurement outcomes.
            - Each T gate acting on a qubit in `q_stab` is replaced by a T-gadget,
              written in terms of a conditional "IF" statement.
            - It then exports the modified circuit (the adaptive Clifford circuit with
              magic state injection) to a new .qasm file.

--------------------------------------------------------------------------------
Author: F.C.R. Peres
Creation date: 02/06/2021
Last updated: 19/07/2022
--------------------------------------------------------------------------------
"""

class Error(Exception):
    """Base class for other exceptions."""
    pass


class RegisterInputError(Error):
    """Exception raised for errors associated with the registers of the input
    quantum circuit.
    """
    def __init__(self, message="Invalid circuit form: registers are not correctly specified!"):
        self.message = message
        super().__init__(self.message)


class NonCliffordTGenInputError(Error):
    """Exception raised when the input circuit has operations other than the
    ones from the {H, S, CNOT, T} generating set.
    """
    def __init__(self, message="Invalid circuit form: illegal operations are being used!"):
        self.message = message
        super().__init__(self.message)



class QuCirc:
    """
    This class represents a given Clifford+T input quantum circuit using an input QASM string.
    
    The QASM string should be in the format produced by qc.qasm() in Qiskit versions < 1.0.
    """
    def __init__(self, qasm_string: str):
        """
        Initialize the QuCirc object with a QASM string.

        Args:
            qasm_string (str): The QASM string representing the quantum circuit.
        """
        self.qasm_string = qasm_string
        # Tokens used for processing the QASM input.
        self.qasm_header = ['OPENQASM ', 'include ']
        self.qasm_registers = ['qreg ', 'creg ']
        self.qasm_allowed_operations = ['h ', 's ', 'cx ', 't ', 'measure ', 'barrier ']
        self.qasm_all = ['OPENQASM ', 'include ', 'qreg ', 'creg ', 'h ', 's ', 'cx ', 't ', 'measure ', 'barrier ']

    def get_circuit_data(self) -> list:
        """
        Process the QASM string and return a list of relevant circuit lines.

        Returns:
            list: Non-blank, non-comment lines (each ending with a newline).
        """
        lines = self.qasm_string.splitlines()
        processed = [line.strip() + "\n" for line in lines if line.strip() and not line.strip().startswith('//')]
        return processed

    def verify_input_circuit(self) -> tuple:
        """
        Verify that the circuit defined by the QASM string is valid and update register names.

        The circuit must have exactly one quantum register and one classical register, at least one
        measurement, and use only allowed operations (h, s, cx, t, measure, barrier). Additionally,
        it must contain at least one T gate (or else it is a simple stabilizer circuit).

        This method also renames registers to 'q_stab' (quantum) and 'c_stab' (classical).

        Returns:
            tuple: (circuit_list, num_qubits, t_count, measurement_count)
                circuit_list (list of str): Updated QASM lines.
                num_qubits (int): Number of qubits (from the qreg declaration).
                t_count (int): Number of T gates.
                measurement_count (int): Number of measurements.
        """
        t_count = 0        # Count T gates.
        mmt_count = 0      # Count measurements.
        nr_qregs = 0       # Count quantum registers.
        circuit_header, circuit_registers, circuit_operations = [], [], []
        circuit_list = self.get_circuit_data()

        for line in circuit_list:
            for token in self.qasm_all:
                if line.startswith(token):
                    if token in self.qasm_header:
                        circuit_header.append(line)
                    elif token in self.qasm_registers:
                        circuit_registers.append(line)
                    elif token in self.qasm_allowed_operations:
                        circuit_operations.append(line)
            if line.startswith('t '):
                t_count += 1
            elif line.startswith('measure '):
                mmt_count += 1
            elif line.startswith('qreg '):
                # Extract the qubit count.
                x = line.partition('[')
                y = x[2].partition(']')
                q_count = int(y[0])
                nr_qregs += 1

        if not (len(circuit_list) == len(circuit_header) + len(circuit_registers) + len(circuit_operations)):
            raise NonCliffordTGenInputError("Circuit contains disallowed operations!")
        elif not (len(circuit_registers) == 2 and nr_qregs == 1):
            raise RegisterInputError("Circuit must have exactly one quantum and one classical register!")

        # Retrieve original register names.
        qreg_name, creg_name = None, None
        for line in circuit_list:
            if line.startswith('qreg '):
                parts = line.split()
                qreg_name = parts[1].split('[')[0]
            elif line.startswith('creg '):
                parts = line.split()
                creg_name = parts[1].split('[')[0]
            if qreg_name and creg_name:
                break

        # Rename registers to standard names.
        updated = []
        for line in circuit_list:
            if qreg_name:
                line = line.replace(f'{qreg_name}[', 'q_stab[')
            if creg_name:
                line = line.replace(f'{creg_name}[', 'c_stab[')
            updated.append(line)
        
        return updated, q_count, t_count, mmt_count

    def msi_circuit(self, msi_clifford_file_loc=None, msi_clifford_file_name=None) -> str:
        """
        Convert the circuit into a gadgetized (adaptive) Clifford circuit with magic state injection.

        The method inserts a new quantum register 'q_magic' and a classical register 'c_magic',
        and replaces each T gate on q_stab with a gadget (using a CNOT, measurement, and conditional S gate).

        Args:
            msi_clifford_file_loc (str, optional): If provided, the gadgetized circuit will also be saved here.
            msi_clifford_file_name (str, optional): If provided, the output file name for the gadgetized circuit.

        Returns:
            str: The gadgetized circuit as a single QASM string.
        """
        cx_count = 0
        hs_count = 0
        circuit_list, q_count, t_count, mmt_count = self.verify_input_circuit()

        for line in circuit_list:
            if line.startswith('cx '):
                cx_count += 1
            elif line.startswith('h ') or line.startswith('s '):
                hs_count += 1

        # Insert auxiliary registers for magic state injection.
        for idx, line in enumerate(circuit_list):
            if line.startswith('creg '):
                circuit_list.insert(idx, f'qreg q_magic[{t_count}];\n')
                circuit_list.insert(idx + 2, f'creg c_magic[{t_count}];\n')
                break

        k = 0
        # Replace each T gate with a gadgetized version.
        for line in circuit_list.copy():
            if line.startswith('t '):
                index = circuit_list.index(line)
                x = line.partition('[')
                y = x[2].partition(']')
                i = y[0]
                new_line = line.replace('t ', 'cx ').replace(';', f',q_magic[{k}];')
                circuit_list[index] = new_line
                circuit_list.insert(index + 1, f'measure q_magic[{k}] -> c_magic[{k}];\n')
                circuit_list.insert(index + 2, f'if(c_magic[{k}]==1) s q_stab[{i}];\n')
                k += 1

        output_qasm = "".join(circuit_list)

        if msi_clifford_file_loc and msi_clifford_file_name:
            if not msi_clifford_file_name.endswith('.qasm'):
                msi_clifford_file_name += '.qasm'
            with open(os.path.join(msi_clifford_file_loc, msi_clifford_file_name), 'w') as f:
                f.write(output_qasm)

        return output_qasm

