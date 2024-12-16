import os
import sys
import time
from itertools import combinations
from pickle import dump, load
from subprocess import check_output
import shutil

import numpy as np
from openbabel import openbabel
from pandas import read_csv
from rdkit import Chem
from rmsd import get_coordinates_xyz, kabsch_rmsd

from ase import Atoms
from ase.io.xyz import write_xyz
from ase.optimize import BFGS
from xtb.ase.calculator import XTB

from scipy.sparse.csgraph import bellman_ford
from scipy.spatial import Voronoi
from scipy.stats import exponnorm

import Algorithm
from Molecule import HiveMolecule

os.environ['OMP_NUM_THREADS'] = '20'

class FailedHive(Exception):
    """Custom exception for failed hive operations."""
    pass

def full_cycle(mol, G_itrs, L_itrs, global_exists, n_confs):
    """
    Perform a full optimization cycle for a molecule using the Hive algorithm.
    1) Initialise Molecule object, get reference information about molecule (RMSD, bond orders)
    2) Initialise G_search and L_search parameters (functions, number of bees, number of iterations, number of permitted failed attempts)
    3) Run global minimum conformation search
    4) Run local minimum conformations search
    5) check whether best conformation suits enуrgy and diversity requirements
    Parameters:
    - mol (HiveMolecule): Object containing information about molecule.
    - G_itrs (int): Number of global minimum conformation search iterations.
    - L_itrs (int): Number of local minimum conformation search iterations.
    - global_exists (str): Path to the global conformation if it exists.
    """
    molecule_preprocessing(mol)
    i = 0
    if mol.dof == 1:
        N_bees = 2
    else:
        N_bees = mol.dof
        
    original_stdout = sys.stdout    
    TIME= time.time()    
    
    """GLOBAL"""
    G_search = global_search
    G_bees = N_bees
    G_itrs = G_itrs
    G_trials = 5
    """LOCAL"""
    L_search = local_search
    L_bees = N_bees
    L_itrs = L_itrs
    L_trials = 5
    smallest_probability = 0.05
    """ ABC ALGORITHM WORK  """
    
    pickle_path = collect_pickle(mol, 'dihedrals', 'mol', 'name', 'mol_bond_info')    
    if global_exists == '':
        global_1, global_2, global_xyz, global_ase = create_global_paths(mol)

        """ GLOBAL SEARCH  """
        FILE = f"./FINAL/{mol.name}/LOGS/{mol.name}_global.log"
        sys.stdout = open(FILE, "w")
        pickle_path = collect_pickle(mol, 'dihedrals', 'mol', 'name', 'mol_bond_info')
        try:
            G_solution, G_population, G_index, G_counter, mol.allbees, mol.allindexes, mol.visited, calculation_counter = run_abc(
                Type='Global',
                fun=G_search,
                numb_bees=G_bees,
                max_itrs=G_itrs,
                max_trials=G_trials,
                ndim=mol.dof,
                last_population=[],
                chosen_index=0,
                last_counter=[],
                allbees=[],
                allindexes=[],
                visited=[],
                glob_exists=False,
                pickle=pickle_path,
                threshold=1
            )
            
            
            molecule_postprocessing(mol, calculation_counter, G_solution, global_1, global_2, global_xyz, fmax=0.1)        

            mol.min_index = 0
            mol.approved_dihedrals.append(G_solution)
            mol.approved_xyzs.append(global_xyz)
            update_pickle(mol, pickle_path, 'approved_xyzs', 'E_list', 'min_index', 'reference_rmsd')
            L_population = calculate_ff_for_global_allbees(mol, L_bees, pickle_path)
            L_index = None
            L_counter = [0]*G_bees

            i += 1
        except FailedHive:
            print(f'{mol.name}: COMPUTATION OF THIS MOLECULE FAILED')
        """ LOCAL SEARCH  """

    else:
        mol.allbees, mol.allindexes, mol.visited, mol.approved_dihedrals, mol.min_index = [], [], [], [], 0
        mol.E_list = [calculate_energy(global_exists)]
        mol.approved_xyzs = [global_exists]
        mol.approved_dihedrals = ['global_dihedrals']
        L_population = []
        L_index = None
        L_counter = [0]*G_bees
        last_trials = []
        update_pickle(mol, pickle_path, 'approved_xyzs', 'E_list', 'min_index', 'reference_rmsd')
    try:
        while len( mol.approved_xyzs) < n_confs and len(mol.rejected_xyzs) < n_confs:
            local_1, local_2, local_xyz, local_ase = create_local_paths(mol, i)   
            update_pickle(mol, pickle_path, 'approved_xyzs', 'E_list', 'min_index')
            FILE = f"./FINAL/{mol.name}/LOGS/{mol.name}_local_{i}.log"
            sys.stdout = open(FILE, "w")
            L_solution, L_population, L_index, L_counter, mol.allbees, mol.allindexes, mol.visited, calculation_counter = run_abc(
                Type='Global',
                fun=L_search,
                numb_bees=L_bees,
                max_itrs=L_itrs,
                max_trials=L_trials,
                ndim=mol.dof,
                last_population=L_population,
                chosen_index=L_index,
                last_counter=L_counter,
                allbees=mol.allbees,
                allindexes=mol.allindexes,
                visited=mol.visited,
                glob_exists=True,
                pickle=pickle_path,
                threshold=1
            )
            molecule_postprocessing(mol, calculation_counter, L_solution, local_1, local_2, local_xyz, fmax=0.1)
            
            EC = EnergyCompare(mol.E_list)
            Cond_1, Cond_2, Cond_3 = calculate_conditions(mol, EC, smallest_probability, local_xyz)
            check_conditions(mol, Cond_1, Cond_2, Cond_3, EC, smallest_probability, L_solution, local_xyz)
            
            i += 1
            
        TIME = time.time()-TIME
        sys.stdout = original_stdout
        with open(f'./FINAL/{mol.name}/PICKLE/{mol.name}.pickle', 'wb') as pick:
            dump(mol, pick)
        for conf in mol.approved_xyzs:
            shutil.copy(conf, f"./FINAL/{mol.name}/ENSEMBLE/{'_'.join(conf.split('/')[-1].split('_')[2:])}")
    except FailedHive:
        print(f'{mol.name}: COMPUTATION OF THIS MOLECULE FAILED')
    return mol.approved_xyzs
        
        
def run_abc(Type, fun, numb_bees, max_itrs, max_trials, ndim, last_population, chosen_index, last_counter, allbees,  allindexes, visited, glob_exists, pickle, threshold):
    """
    Run the Artificial Bee Colony (ABC) algorithm.

    Parameters:
    - Type (str): Type of search ('Global' or 'Local').
    - fun (function): Function converting vector of molecule dihedral angles to Energy and Fitness Function
    - numb_bees (int): Number of bees, simultaneously flying above the PES
    - max_itrs (int): Number of algorithm iterations
    - max_trials (int): Maximum number of trials.
    - ndim (int): Permitted number of failed improvement attempts
    - last_population (list): Population of bees in the end of last algorithm work.
    - chosen_index (int): Index of the best bee of last algorithm work.
    - last_counter (list): Counter of failed improvement attempts during last algorithm work.
    - allbees (list): All vectors that lead to an unbroken molecule.
    - allindexes (list): Indexes of 'allbees' list which correspond to best conformations on every algorithm work
    - visited (list): All vectors found, including those leading to broken molecules
    - glob_exists (bool): Determins type of search. "Global" if False and "Local" if True.
    - pickle (str): Path to pickle file containing required molecule parameteres.
    - threshold (int): Minimum difference between the components of two vectors

    Returns:
    - model.solution (list): Best vector obtained during current iteration
    - model.population (list): Population of bees in the end of current algorithm work.
    - model.best_index (int): Index of the best bee of current algorithm work.
    - model.counter (list): Counter of failed improvement attempts during current algorithm work.
    - model.allbees (list): 'allbees' list including new index got after current algorithm work
    - model.allindexes (list): 'allindexes' list including new index got after current algorithm work
    - model.visited (list): 'visited' list including new vectors got after current algorithm work
    - model.calculation_counter (int): Number of energy calculator calls.
    """
    try:
        model = Algorithm.BeeHive(
                                Type=Type,
                                lower=[-180] *ndim,
                                upper=[180]*ndim,
                                fun=fun,
                                numb_bees=numb_bees,
                                max_itrs=max_itrs,
                                max_trials=max_trials,
                                threshold=threshold,
                                last_population=last_population,
                                chosen_index=chosen_index,
                                last_counter=last_counter,
                                allbees=allbees,
                                allindexes=allindexes,
                                visited=visited,
                                pickle=pickle,
                                glob_exists=glob_exists
                                )

    except Algorithm.FailedBee:
        raise FailedHive

    try:
        model.run()
        return model.solution, model.population, model.best_index, model.counter, model.allbees, model.allindexes, model.visited, model.calculation_counter
    except Algorithm.FailedBee:
        raise FailedHive

def calculate_conditions(mol, EC, p, local_xyz):
    """
    Calculate conditions for the optimization process.

    Parameters:
    - mol (HiveMolecule): Object containing information about molecule.
    - EC (list): Thermodynamic probabilites of existence of conformations in an equilibrium system.
    - p (float): Threshold of minimim thermodynamic probability which is suitable for for enesemble.
    - local_xyz (str): Local XYZ file path.

    Returns:
    - Cond_1 (bool): Shows whether the found conformation satisfies the 1st selection criteria
    - Cond_2 (bool): Shows whether the found conformation satisfies the 2nd selection criteria
    """
    Cond_1 = all(conf > p for conf in EC)
    if mol.dof < 2:
        Cond_2 = True
    else:
        Cond_2 = calculate_PPE_path(mol)
    Cond_3 = calculate_final_rmsd(mol, local_xyz)
    return Cond_1, Cond_2, Cond_3
        
def global_search(dihedral_values, pickle_file):
    """
    Perform global search for the molecule:
    0) initialising additional parameters from pickle
    1) transforming list of dihedral angles to XYZ and MOL file
    2) checking conformation bond orders
    3) if conformation bond orders are equal to required bond orders, calculating conformation energy

    Parameters:
    - dihedral_values (list): Dihedral angles.
    - pickle_file (str): Path to pickle file containing required molecule parameteres.

    Returns:
    - Energy (float): Calculated Energy
    - Fitness (float): Fitness value
    """ 
    with open(pickle_file, 'rb') as pickle_file:
        pdict= load(pickle_file)    
    mol, name, dihedrals, mol_bond_info = pdict['mol'], pdict['name'], pdict['dihedrals'], pdict['mol_bond_info']
    
    file = f'./FINAL/{name}/LOGS/{name}.xyz'
    molfile = f'./FINAL/{name}/LOGS/{name}.mol'
    semiopt_asefile = f'./FINAL/{name}/LOGS/ASE_{name}.xyz'

    rdmol_to_xyz(mol, dihedrals, dihedral_values, file)
    to_mol(file, molfile)
    
    if check_bonds(get_mol_bond_info(molfile), mol_bond_info) is True:
        Energy = optimization(file, semiopt_asefile, fmax=0.5)
        Fitness = -Energy
        print(f'{np.around(dihedral_values, decimals = 4)}     {Energy:0.4f}')
        return Energy, Fitness
    else:
        return 0, 0

def local_search(dihedral_values, pickle_file):
    """
    Perform local search for the molecule.
    0) initialising additional parameters from pickle
    1) transforming list of dihedral angles to XYZ and MOL file
    2) checking conformation bond orders
    3) if conformation bond orders are equal:
        a) calculating conformation Energy
        b) calculating RMSD between current conformation and conformations approved before
    4) calculating Fitness Fuтсtion which depends on Energy and RMSD

    Parameters:
    - dihedral_values (list): Dihedral angles.
    - pickle_file (str): Path to pickle file containing required molecule parameteres.

    Returns:
    - Energy (float): Calculated Energy
    - Fitness (float): Fitness value
    """ 
    with open(pickle_file, 'rb') as pickle_file:
        pdict= load(pickle_file)
    mol, name, dihedrals, mol_bond_info, approved_xyzs, E_list, min_index, reference_rmsd = pdict['mol'], pdict['name'], pdict['dihedrals'], pdict['mol_bond_info'], pdict['approved_xyzs'], pdict['E_list'], pdict['min_index'], pdict['reference_rmsd'],
    
    file = f'./FINAL/{name}/LOGS/{name}.xyz'
    molfile = f'./FINAL/{name}/LOGS/{name}.mol'
    asefile = f'./FINAL/{name}/LOGS/ASE_{name}.xyz'
    xyzfile = f'./FINAL/{name}/LOGS/XYZ_{name}.xyz'

    rdmol_to_xyz(mol, dihedrals, dihedral_values, file)
    to_mol(file, molfile)
    
    if check_bonds(get_mol_bond_info(molfile), mol_bond_info) is True:
        Energy = optimization(file, asefile, fmax=0.5)
        redact_ase(asefile, xyzfile)
        rmsd_min_A = compare_rmsd(approved_xyzs, xyzfile)
        rmsd_part = rmsd_exponnorm_fitness(rmsd_min_A, reference_rmsd)
        E_part = E_calculate_fitness(Energy, E_list[min_index])
        FF = float(rmsd_part*E_part)
        
        print(f'{np.around(dihedral_values, decimals = 3)}')
        print('\nE:', np.around(Energy, decimals=4), '\nE_part:', np.around(E_part, decimals=4), '\nrmsd:',np.around(rmsd_min_A, decimals=4), '\nrmsd_part:',np.around(rmsd_part, decimals=4))
        print(f'FF:{np.around(FF, decimals = 4)}')
        return Energy,  FF

    else:
        return 0, 0        

def molecule_preprocessing(mol):
    """
    Preprocess the molecule before optimization.
    1) get bond orders for some default conformation
    2) calculate reference RMSD to build Fitness Function based on it
    
    Parameters:
    - mol (HiveMolecule): Object containing information about molecule.
    """
    create_common_folders(mol)
    reference_xyz = f'./FINAL/{mol.name}/LOGS/{mol.name}_reference.xyz'
    reference_mol = f'./FINAL/{mol.name}/LOGS/{mol.name}_reference.mol'

    Chem.rdmolfiles.MolToXYZFile(mol.mol, reference_xyz)
    to_mol(reference_xyz, reference_mol)
    mol.mol_bond_info = get_mol_bond_info(reference_mol)
    mol.reference_rmsd = calculate_reference_rmsd(mol, mol.dof)
    mol.calculation_counter = 0
    
def molecule_postprocessing(mol, calculation_counter, solution, conf_1, conf_2, conf_xyz, fmax):
    """
    Postprocess the molecule after optimization.
    Calculate the energy of approved conformation with less gradient descent step

    Parameters:
    - mol (HiveMolecule): Object containing information about molecule.
    - calculation_counter (int): Calculation counter.
    - solution (list): Solution dihedral values.
    - conf_1 (str): Path to the first configuration file.
    - conf_2 (str): Path to the second configuration file.
    - conf_xyz (str): Path to the XYZ configuration file.
    - fmax (float): Maximum force.
    """
    mol.calculation_counter += calculation_counter
    rdmol_to_xyz(mol.mol, mol.dihedrals, solution, conf_1)
    E = optimization(conf_1, conf_2, fmax=0.1)
    mol.E_list.append(E)
    redact_ase(conf_2, conf_xyz)
    
def rdmol_to_xyz(mol, dihedrals, solution, path):
    """
    Convert RDKit molecule to XYZ format.

    Parameters:
    - mol (RDKit Mol): The molecule.
    - dihedrals (list): List of dihedral angles.
    - solution (list): Solution dihedral values.
    - path (str): Path to save the XYZ file.
    """
    for dihed, idx in zip(solution, dihedrals):
        Chem.rdMolTransforms.SetDihedralDeg(mol.GetConformer(), *idx, dihed)
    Chem.rdmolfiles.MolToXYZFile(mol, path)

def to_mol(infile, outfile):
    """
    Convert XYZ file to MOL format using OpenBabel.

    Parameters:
    - infile (str): Path to the input XYZ file.
    - outfile (str): Path to save the output MOL file.
    """
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("xyz", "mol2")
    mol = openbabel.OBMol()
    conv.ReadFile(mol, infile)
    conv.WriteFile(mol, outfile)
    
def check_bonds(list1, list2):
    """
    Check whether two lists of bonds are same.

    Parameters:
    - list1 (list): First list of bonds.
    - list2 (list): Second list of bonds.

    Returns:
    - bool: True if same, False otherwise.
    """
    if len (list1) == len(list2):
        flags = []
        for i in list1:
            flag = False
            for j in list2:
                if (j[0][0] == i[0][0] and j[0][1] == i[0][1]) or (j[0][0] == i[0][1] and j[0][1] == i[0][0]) and (j[1] == i[1]):
                    flag = True
            flags.append(flag)
        return all(flags)
    else:
        return False
    
def make_atoms_from_xyz_file(xyz_file):
    """
    Create ASE Atoms object from XYZ file.

    Parameters:
    - xyz_file (str): Path to the XYZ file.

    Returns:
    - atoms (Atoms): ASE Atoms object.
    """
    with open (xyz_file) as data:
        data = [x.split() for x in data]
        symbols = [x[0] for x in data if len(x) == 4 and '=' not in x]
        positions = [(float(x[1]), float(x[2]), float(x[3])) for x in data if len(x) == 4 and '=' not in x]
    atoms = Atoms(symbols, positions)
    return atoms

def calculate_energy(xyz_file):
    """
    Calculate the energy of a molecule from an XYZ file.

    Parameters:
    - xyz_file (str): Path to the XYZ file.

    Returns:
    - e (float): Energy.
    """
    atoms = make_atoms_from_xyz_file(xyz_file)
    atoms.calc = XTB(method="GFN2-xTB")
    e = atoms.get_potential_energy()  
    return e    
    
def optimization(xyz_file, ase_file, fmax):
    """
    Optimize the molecule using ASE and XTB.

    Parameters:
    - xyz_file (str): Path to the input XYZ file.
    - ase_file (str): Path to save the optimized ASE file.
    - fmax (float): Maximum force.

    Returns:
    - e (float): Energy.
    """
    atoms = make_atoms_from_xyz_file(xyz_file)
    atoms.calc = XTB(method="GFN2-xTB")
    trajectory = ase_file.split('.')[0]+'.traj'
    atoms_bfgs = BFGS(atoms, trajectory=trajectory)
    atoms_bfgs.run(fmax=fmax)
    atoms_opt = ase_file
    write_xyz(atoms_opt, atoms)
    e = atoms.get_potential_energy()  
    return e
    
def redact_ase(ase_file, new_file):
    """
    Redact ASE file to remove unnecessary information.

    Parameters:
    - ase_file (str): Path to the input ASE file.
    - new_file (str): Path to save the redacted file.
    """
    with open(f'{new_file}', 'w') as new_file, open(f'{ase_file}', 'r') as ase_file:
        for line in ase_file:
            if len(line.split()) == 8:
                new_file.write(' '.join(line.split()[:4]))
                new_file.write('\n')
            else:
                new_file.write(line)
        new_file.close()
        ase_file.close()
    
def compare_rmsd(approved, candidate):
    """
    Compare RMSD of candidate with approved conformations.
    Choose minimum RMSD

    Parameters:
    - approved (list): List of paths to approved XYZ files.
    - candidate (str): Path to the candidate XYZ file.

    Returns:
    - rmsd_min (float): Minimum RMSD.
    """
    rmsd_relative = []
    for i in approved:
        rmsd_wH = calculate_rmsd(i, candidate)
        rmsd_relative.append(rmsd_wH)
    rmsd_min = min(rmsd_relative)
    return rmsd_min

def calculate_rmsd(file_1, file_2):
    """
    Calculate RMSD between two XYZ files.

    Parameters:
    - file_1 (str): Path to the first XYZ file.
    - file_2 (str): Path to the second XYZ file.

    Returns:
    - float: RMSD.
    """
    conf_1, conf_2 =  get_coordinates_xyz(file_1)[1],  get_coordinates_xyz(file_2)[1]

    return kabsch_rmsd(conf_1, conf_2, translate=True)

def rmsd_exponnorm_fitness(x, reference_rmsd):
    """
    Calculate RMSD fitness using exponential normal distribution.

    Parameters:
    - x (float): RMSD value.
    - reference_rmsd (float): Reference RMSD.

    Returns:
    - у (float): Fitness value.
    """
    y = exponnorm.cdf(x, K=1, loc=reference_rmsd, scale=2)
    return y

def E_calculate_fitness(x, E_glob):
    """
    Calculate energy fitness.

    Parameters:
    - x (float): Energy value.
    - E_glob (float): Global energy.

    Returns:
    - float: Fitness value.
    """
    if x < E_glob:
        return 1
    else:
        return (1/(1+10*(x-E_glob)))

def EnergyCompare(Enlist):
    """
    Сalculate Thermodynamic probabilites of existence of conformations in an equilibrium system

    Parameters:
    - Enlist (list): List of energies (eV).

    Returns:
    - probability (list): List of thermodynamic probabilities.
    """
    print('Energies:', np.around(Enlist, decimals=4))
    if len (Enlist) == 1:
        return [1]
    neigh_probs = []
    for i in range(len(Enlist)-1):
        neigh_probs.append(np.exp(((Enlist[i+1]-Enlist[i]))*96.48535*1000/8.314/298.15))
    neigh_probs.append(1)
    for i in range(len(neigh_probs)-1, 0, -1):
        neigh_probs[i-1] *= neigh_probs[i]
    probability = [i/sum(neigh_probs) for i in neigh_probs]
    return probability

def delete_highest_conformations(mol, EC, p):
    """
    Delete highest conformations based on energy probabilities.

    Parameters:
    - mol (HiveMolecule): Object containing information about molecule.
    - EC (list): Energy comparison list.
    - p (float): Probability.

    Returns:
    - min_index (int): Index of conformation with minimim energy.
    """
    highest_indexes = []
    for i in range(len(EC)):
        if EC[i] < p:
            highest_indexes.append(i)
            
    print('E_list:', mol.E_list)
    print('approved_xyzs', mol.approved_xyzs)
    print('approved_dihedrals', mol.approved_dihedrals)
    print('highest_indexes', highest_indexes)
    for i in highest_indexes[::-1]:
        print(i)
        EC.pop(i)
        mol.E_list.pop(i)
        mol.allindexes.pop(i)
        mol.rejected_xyzs.append(mol.approved_xyzs.pop(i))
        mol.rejected_dihedrals.append(mol.approved_dihedrals.pop(i))
        print('E_list:', mol.E_list, 'appoved_xyzs:', mol.approved_xyzs,'all_indexes:', mol.allindexes)
    min_index = EC.index(max(EC))
    return min_index

def check_conditions(mol, Cond_1, Cond_2, Cond_3, EC, p, L_solution, local_xyz):
    """
    Сhecking the conditions for conformations to enter the ensemble

    Parameters:
    - mol (HiveMolecule): Object containing information about molecule.
    - Cond_1 (bool): Energy condition.
    - Cond_2 (bool): Having a transition point condition.
    - Cond_3 (bool): Having nonzero RMSD with approved optimised conformations
    - EC (list): Energy comparison list.
    - p (float): Probability.
    - L_solution (list): Local solution dihedral values.
    - local_xyz (str): Local XYZ file path.
    """
    if all((Cond_1,Cond_2, Cond_3)):
        mol.approved_dihedrals.append(L_solution)
        mol.approved_xyzs.append(local_xyz)

    else:
        if Cond_1 is False:
            if EC[-1] < p:
                mol.E_list.pop()
                mol.rejected_dihedrals.append(L_solution)
                mol.rejected_xyzs.append(local_xyz)
                mol.allindexes.pop()
            else:
                mol.min_index = delete_highest_conformations(mol, EC, p=p)
                mol.approved_dihedrals.append(L_solution)
                mol.approved_xyzs.append(local_xyz)
        else:
                mol.E_list.pop()
                mol.rejected_dihedrals.append(L_solution)
                mol.rejected_xyzs.append(local_xyz)
                mol.allindexes.pop()

def calculate_final_rmsd(mol, candidate):
    for i in mol.approved_xyzs:
        rmsd = calculate_rmsd(i, candidate)
        if rmsd < 0.15:
            print('2: Kabsch RMSD: similar to another confrmation')
            return False
    print('2: Kabsch RMSD: different to other conformations')
    return True                

def calculate_ff_for_global_allbees(mol, num_bees, pickle_path):
    """
    Calculate fitness function for all bees to determine the initial hive for local search.

    Parameters:
    - mol (HiveMolecule): Object containing information about molecule.
    - num_bees (int): Number of bees.
    - pickle_path (str): Path to pickle file containing required molecule parameteres.

    Returns:
    - best_bees (list): Best bees for local search.
    """
    for bee in mol.allbees:
        bee.value, bee.fitness = local_search(bee.vector, pickle_path)
    allfitnesses = [(i, np.around(mol.allbees[i].fitness, decimals=5)) for i in range(len(mol.allbees))]
    print('allfitnesses:', allfitnesses)
    sorted_allbees = sorted(allfitnesses, key=lambda x: x[1], reverse=True)
    sorted_indexes = [x[0] for x in sorted_allbees[:num_bees]]
    best_bees = [mol.allbees[i] for i in range(len(mol.allbees)) if i in sorted_indexes]
    print('BEST BEES FOR LOCAL SEARCH:\n', np.around([bee.vector for bee in best_bees], decimals=4))
    return best_bees

def create_path_to(array, index_in, index_out, allbees):
    """
    Create a path between two nodes in the graph.

    Parameters:
    - array (list): Predecessor array.
    - index_in (int): Index of the starting node.
    - index_out (int): Index of the ending node.
    - allbees (list): List of all bees.

    Returns:
    - list: Path.
    """
    path = []
    node = index_out
    while node != -9999:
        path.append(node)
        node = array[node]
    if path[-1] == index_in:
        print('Path is fully built')
        print('Bees along the path:')
        for i in path:
            print('    ', np.around(allbees[i], decimals=4))
        print(path)
        return path[::-1]
    else:
        print('Path cannot be built')
        path.append(-9999)
        print(path)
        return path[::-1]

def find_the_transition(mol, pair, ind_1, ind_2):
    """
    Find the conformation with higher energy between two conformations.

    Parameters:
    - mol (HiveMolecule): Object containing information about molecule.
    - pair (tuple): Pair of paths.
    - ind_1 (int): Index of the first conformation in 'allbees' list.
    - ind_2 (int): Index of the second conformation in 'allbees' list.

    Returns:
    - bool: True if transition found, False otherwise.
    """
    curve_counter = 0
    for a in pair[0][1:-1]:
        if mol.allbees[a].value > mol.E_list[ind_1] and mol.allbees[a].value > mol.E_list[ind_2]:
            curve_counter += 1
            print(f"There's a transition in conformation {a}: { np.around(mol.allbees[a].vector, decimals = 4)}")
            break
    for a in pair[1][1:-1]:
        if mol.allbees[a].value > mol.E_list[ind_1] and mol.allbees[a].value > mol.E_list[ind_2]:
            curve_counter += 1
            print(f"There's a transition in conformation {a}: { np.around(mol.allbees[a].vector, decimals = 4)}")
            break
    if curve_counter < 2:
        return False
    else:
        return True

def calculate_PPE_path(mol):
    """
    Calculate the Potential Path Energy (PPE) path for the molecule.
    1) сalculate Voronoi polyhedra on the space of found conformations
    2) calculate connectivity matrix based on Voronoi diagram
    3) build a path between the new conformation and all previously selected conformations
    4) check if there are conformations along the path that have higher energy than the conformations at the ends

    Parameters:
    - mol (HiveMolecule): Object containing information about molecule.

    Returns:
    - bool: True if path found, False otherwise.
    """
    alldihedrals = [bee.vector for bee in mol.allbees]
    voronoi = Voronoi(alldihedrals)
    allbees_matrix = np.zeros((len(mol.allbees), len(mol.allbees)))
    
    for pair in voronoi.ridge_points:
        i, j = pair[0], pair[1]
        allbees_matrix[i][j] = mol.allbees[i].value - mol.allbees[j].value
        allbees_matrix[j][i] = -allbees_matrix[i][j]
        
    _, predecessors = bellman_ford(csgraph=allbees_matrix, indices=mol.allindexes, directed=True, return_predecessors=True)
    allindexes_combinations = [i for i in combinations(mol.allindexes, 2) if mol.allindexes[-1] in i]
    print('3: allindexes_combinations', allindexes_combinations)
    arrays = []
    allvectors = [bee.vector for bee in mol.allbees]
    
    for comb in allindexes_combinations:
        ind_1, ind_2 = mol.allindexes.index(comb[0]), mol.allindexes.index(comb[1])
        print(f'creating path between {comb[1]} and {comb[0]} conformations:')
        path_1 = create_path_to(predecessors[mol.allindexes.index(comb[1])], comb[1], comb[0], allvectors)
        print('looking for a reversed path:')
        path_2 = create_path_to(predecessors[mol.allindexes.index(comb[0])], comb[0], comb[1], allvectors)
        if len(path_1) == 2 or len(path_2) == 2:
            return False
        else:
            print('looking for a transition:')
            print(f'Energies along path 1: {np.around([mol.allbees[a].value for a in path_1][1:-1], decimals=4)}')
            print(f'Energies along path 2: {np.around([mol.allbees[a].value for a in path_2][1:-1], decimals=4)}')
            arrays.append(find_the_transition(mol, (path_1, path_2), ind_1, ind_2))
        if all(arrays) is True:
            print("transition conformations found ")
        else:
            print("transition conformations were not found ")
            return False
    return True

def collect_pickle(mol, *args):
    """
    Collect important parameters and save the pickle file for the molecule.

    Parameters:
    - mol (HiveMolecule): Object containing information about molecule.
    - *args: Additional arguments required for global or local search.

    Returns:
    - pickle path (str): Path to the pickle file.
    """
    pickle_dict = {}
    for arg in args:
        pickle_dict[arg] = mol.__dict__[arg]
    pickle_path = f'./FINAL/{mol.name}/LOGS/{mol.name}.pickle'
    with open(pickle_path, 'wb') as pickle:
        dump(pickle_dict, pickle)
    return pickle_path

def update_pickle(mol, pickle_path, *args):
    """
    Update the pickle file with new information.

    Parameters:
    - mol (HiveMolecule): Object containing information about molecule.
    - pickle_path (str): Path to pickle file containing required molecule parameteres.
    - *args: Additional arguments required for global or local search.
    """
    with open(pickle_path, 'rb') as pickle:
        pickle_dict = load(pickle)
    for arg in args:
        pickle_dict[arg] = mol.__dict__[arg]
    with open(pickle_path, 'wb') as pickle:
        dump(pickle_dict, pickle)

def calculate_reference_rmsd(mol, num_dihedrals):
    """
    Calculate the reference RMSD for the molecule to build RMDS Fitness Function

    Parameters:
    - mol (HiveMolecule): Object containing information about molecule.
    - num_dihedrals (int): Number of dihedrals.

    Returns:
    - float: Minimum RMSD.
    """
    ref_1 = f'./FINAL/{mol.name}/LOGS/{mol.name}_rmsd_1.xyz'
    dihs_1 = np.zeros((num_dihedrals))

    ref_2 = f'./FINAL/{mol.name}/LOGS/{mol.name}_rmsd_2.xyz'
    dihs_2 = np.zeros((num_dihedrals))
    dihs_2[0] = 180

    ref_3 = f'./FINAL/{mol.name}/LOGS/{mol.name}_rmsd_3.xyz'
    dihs_3 = np.zeros((num_dihedrals))
    dihs_3[-1] = 180

    rdmol_to_xyz(mol.mol, mol.dihedrals, dihs_1, ref_1)
    rdmol_to_xyz(mol.mol, mol.dihedrals, dihs_2, ref_2)
    rdmol_to_xyz(mol.mol, mol.dihedrals, dihs_3, ref_3)

    time_rmsd = time.time()
    rmsd_12 = float(calculate_rmsd(ref_1, ref_2))
    rmsd_13 = float(calculate_rmsd(ref_1, ref_3))
    time_rmsd = time.time() - time_rmsd
    return np.min([rmsd_12, rmsd_13])

def get_mol_bond_info(molfile):
    """
    Get the bond information from the MOL file.

    Parameters:
    - molfile (str): Path to the MOL file.

    Returns:
    - list: List of conformation bonds.
    """
    with open(molfile, 'r') as file:
        file_list = file.readlines()
        flag = False
        bonds = []
        for line in file_list:
            if flag:
                bonds.append([(line.split()[1], line.split()[2]), line.split()[3]])
            elif '@<TRIPOS>BOND' in line:
                flag = True
    return bonds

def create_common_folders(mol):
    """
    Create common folders for storing logs, ASE files, bees, and pickles.
    """
    paths_list = ['LOGS','ASE','BEES/OPT', 'BEES/UNOPT', 'PICKLE', 'ENSEMBLE']
    for i in paths_list:
        os.makedirs(name=f"./FINAL/{mol.name}/{i}", exist_ok=True)

def create_global_paths(mol):
    """
    Create paths for global optimization files.

    Parameters:
    - mol (HiveMolecule): Object containing information about molecule.

    Returns:
    - tuple: Paths to different formats of global conformation file.
    """
    global_1 = f"./FINAL/{mol.name}/BEES/UNOPT/{mol.name}_global.ase"
    global_2 = f"./FINAL/{mol.name}/BEES/OPT/ASE_{mol.name}_global.ase"
    global_xyz = f"./FINAL/{mol.name}/BEES/OPT/XYZ_{mol.name}_global.xyz"
    global_ase = f"./FINAL/{mol.name}/ASE/{mol.name}_global"   
    
    return global_1, global_2, global_xyz, global_ase

def create_local_paths(mol, i):
    """
    Create paths for local optimization files.

    Parameters:
    - mol (HiveMolecule): Object containing information about molecule.
    - i (int): Number of local conformation.

    Returns:
    - tuple: Paths to different formats of local conformation file.
    """
    local_1 = f"./FINAL/{mol.name}/BEES/UNOPT/{mol.name}_local_{i}.ase"
    local_2 = f"./FINAL/{mol.name}/BEES/OPT/ASE_{mol.name}_local_{i}.ase"
    local_xyz = f"./FINAL/{mol.name}/BEES/OPT/XYZ_{mol.name}_local_{i}.xyz"
    local_ase = f"./FINAL/{mol.name}/ASE/{mol.name}_local_{i}"    
    
    return local_1, local_2, local_xyz, local_ase