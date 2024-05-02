## Welcome to Polytope Simulation Project

#### Introduction
This project is about simulating quantum mechanics using a polytope based simulation algorithm. For further information,
see this paper: [A hidden variable model for universal quantum computation with magic states on qubits.
](https://arxiv.org/abs/2004.01992)

#### Installing Dependencies for User
To install the dependencies for a user, run the following command in the terminal:
```bash
    pip install -r requirements.txt
```

#### Installing Dependencies for Developer
To install the dependencies for a developer, run the following command in the terminal:
```bash
    pip install -r requirements-dev.txt
```


#### Creating the Documentation
To create the documentation, first one should change the directory to the `docs` folder:
```bash
    cd docs
```
Then, run the following command in the terminal:
```bash
    make html
```

**Note:** If a new file is added into .rst files need to be created in the `docs` folder. To do that, one can create a `.rst` file in the `docs` folder and add its name into `modules.rst` file. 