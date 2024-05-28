# Quantum circuit conversion

In this repository we develop code for converting a quantum computation in the circuit-based model to a measurement-based quantum computation (MBQC) scheme. Typically MBQC models begin with a quantum state containing a nonclassical resource (e.g., entanglement) and the quantum computation proceeds by successive measurements on the resource state, consuming the nonclassical resource in the process.

There are several models of measurement-based quantum computation (MBQC). There is the one-way quantum computer first developed by Raussendorf and Briegel [https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.86.5188] and more recently, quantum computation with magic states (QCM) [https://arxiv.org/abs/quant-ph/0403025], which was later refined to Pauli-based computation (PBC) by Bravyi-Smith-Smolin [https://arxiv.org/abs/1506.01396]. In many, if not all cases, the measurement-based scheme requires an adaptive set of measurements and classical side processing is needed to choose the subsequent measurements so that the desired quantum algorithm is implemented. Since the complexity of quantum computation on a classical computer In each case the conversions should be polynomial in time.

Our baseline PBC code draws from [https://github.com/fcrperes/CompHybPBC] which is based on the work of F.C. Peres and E.F. Galvao in [https://arxiv.org/abs/2203.01789]. For instructions and description on that code see [https://github.com/fcrperes/CompHybPBC].
-------------------------------------------------------------------------------


**Prerequisites:** This is for the PBC by F.C. Peres code:
* Python 3.7.10;
* numpy 1.20.1;
* matplotlib 3.3.4;
* 'qiskit-terra': '0.17.4', 'qiskit-aer': '0.8.2', 'qiskit-ignis': '0.6.0',
'qiskit-ibmq-provider': '0.13.1', 'qiskit-aqua': '0.9.1', 'qiskit': '0.26.2',
'qiskit-nature': None, 'qiskit-finance': None, 'qiskit-optimization': None,
'qiskit-machine-learning': None


**Brief description:** The modules supplied allow us to execute two different
tasks (as described in our pre-print [https://arxiv.org/abs/2203.01789]):
* Task 1: Efficient circuit compilation and weak simulation; to carry out this
task one should open the `Main.py` module, change the number of virtual qubits,
`vq`, to 0 and adjust the parameters of the function `cc.run_pbc` as desired;
* Task 2: Hybrid computation using virtual qubits and approximate strong simu-
lation with maximum relative error ϵ; to perform this task one should open the
`Main.py` module, change the number of virtual qubits, `vq`, to a number greater
than 0, and adjust the parameters of the function `cc.hybrid_pbc` as desired.

 
**Use instructions**:
1. Make sure that the four Python modules (`Main.py`, `input_prep_t1.py`,
`input_prep_t2.py`, and `c_and_c.py`) are all in the same directory in your
computer;
2. Copy the input (.qasm) toy files supplied in this repository into a folder
in your computer (or make your own files with suitable Clifford+T quantum
circuits);
3. The files output by the code will be saved to a folder named "output" which
will be found inside the folder where you have placed the input file or files;
4. In the `Main.py` file change the location of the input files to point to the
correct location in your computer; change also the name of the input file(s)
appropriately, and choose the desired name for the different output files.
Adjust the parameters of `cc.run_pbc` or `cc.hybrid_pbc` depending on the
simulation you want to run;
5. Open a terminal window in the directory where you have placed the Python
modules and run the command: `python Main.py`;
6. Check the output files at the location that you have selected.