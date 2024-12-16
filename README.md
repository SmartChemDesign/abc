# abc
Collecting small organic molecule ensembles using Artificial Bee Colony Algorithm (ABC)


An Artificial Bee Colony Algorithm (ABC) (https://link.springer.com/article/10.1007/S10898-007-9149-X)
implementation for conformational search of small organic molecules

Based on Hive Python Package implementing ABC Algorithm in Python (https://doi.org/10.5281/zenodo.1004592%7D)


## Files decriptions
- **Run.py** - a file that receives input information about the molecule, the type of search and the number of iterations
- **Molecule.py** - a file that translates smiles or the path to the file containing the molecule into an object containing all the information about the conformations of the molecule during the calculation
- **Cycle.py** - a file that contains a call to global and local conformational searches, calculates the necessary functions and compares the conformations with each other
- **Algorithm.py** - a file that contains the Artificial Bee Colony Algorithm itself, a single one for local and global searches

**Necessary parameters**
- **"input"** - choose between "molfile" and "smiles"
- **"molecule"** - string  contains smiles or path to .mol-file

**additional parameters**
- **"--n_confs"** - number of conformations in enseble
- **"--global_iterations"** - number of ABC iterations of global conformation search
- **"--local_iterations"** - number of ABC iterations of local conformation search
- **"--global_exists"** - path to .xyz-file containing global minimum conformation of required molecule


## How to install 
1. You should create conda environment. If you don't have conda, go to [install_conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Copy abc_conformations.yml file to your machine
3. Type in linux terminal
```
conda env create -f abc_conformations.yml
```
then, activate environment
```
conda activate abc_conformations
```

## How to use

1. First, you should copy abc_conformations directory to your computer
2. Write in command line:

```
python3 Run.py {input} {molecule} --n_confs XX --global_iterations XX --local_iterations XX --global-exists <path/to/global/xyzfile>
```

3. In the folder where the package files are located, the following folder system will be created:
- FINAL
    - molecule-name
        - LOGS
        - BEES
            - OPT
            - UNOPT
        - ASE
        - PICKLE
		- ENSEMBLE
		
- **FINAL** - common folder for all molecules you want to calculate
- **LOGS** - folder with temporary files and logs of ABC Algorithm work for every conformation
- **BEES** - all calculated conformations
   - 	**UNOPT** - conformations got after ABC Algorithm work    
   - 	**OPT** - optimized conformations which may be included in ensembe
- **PICKLE** - folder with .pickle-file containing all molecule data after calculation
- **ENSEMBLE** - folder with final ensemble 

4. When calculation is over, approved conformations will be located in ```./FINAL/molecule-name/ENSEMBLE directory```
