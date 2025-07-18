## Welcome to CNCSim Project

#### Introduction
This project is about classical simulation of quantum algorithms with qubits. Quantum computing with magic states (QCM) is a universal model of quantum computing that is an alternative to the usual circuit based model and is driven by stabilizer operations acting on a "magic" input state [[Bravyi-Kitaev, 2004]](https://arxiv.org/abs/quant-ph/0403025).


Here we classically simulate QCM using methods based on sampling from a quasi-probability distribution. These are objects that behave like probability distributions except that their values are allowed to be negative, which can be understood as the onset of "quantumness" into the quantum computation. Howard and Campbell [[Howard-Campbell, 2016]](https://arxiv.org/abs/1609.07488) introduced a quasi-probability representation based on stabilizer states and the cost of classical simulation was controlled by the optimal amount of negativity, called the robustness of magic.

Our simulation approach is based on maximal Closed Non-Contextual (CNC) operators [[Raussendorf et al., 2020]](https://arxiv.org/abs/1905.05374), objects which subsume stabilizer states, therefore promising less negativity and more efficient classical simulation. Central to our implementation is a novel tableau structure, called the *CNC tableau* [[Ipek et al., 2025]](https://arxiv.org/abs/2506.04033). This structure efficiently represents (in the number of qubits, $n$) CNC operators in a tableau of size $\mathcal{O}(n^2)$, much in the same way that the Aaronson-Gottesman tableau [[Aaronson-Gottesman, 2004]](https://arxiv.org/abs/quant-ph/0406196) efficiently represents stabilizer states. The update rules of the CNC tableau under application of Clifford unitaries $\mathcal{O}(n)$ and measurement of Pauli observables $\mathcal{O}(n^2)$ are also efficient.

Maximal CNC operators are also vertices of a class of polytopes introduced in [[Zurel et al., 2020]](https://arxiv.org/abs/2004.01992) for classical simulation of universal quantum computation in the QCM model. An interesting aspect of this approach is that negativity disappears entirely, raising the question of what quantum property is responsible for the presumed hardness of classical simulation. Our study of CNC based simulation in this repository is a first step towards understanding this open problem.

#### Installing Dependencies for User
To install the dependencies for a user, run the following command in the terminal:
```bash
    pip install -r requirements.txt
```


## Patent Pending Notice
This repository contains material related to our arXiv preprint arXiv:2506.04033. Please note that the technology described herein is subject to a pending patent (application number: 2025/007146). Licensed under MIT, which does not grant patent rights. 

For inquiries: cihan.okay@bilkent.edu.tr.
