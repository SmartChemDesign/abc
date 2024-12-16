from operator import itemgetter

import pandas as pd
import numpy as np

import rmsd
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms


class HiveMolecule:
    """
    class HiveMolecule contains functionality to test
    ABC method on the reference molecule
    """
    def __init__(self, input_type, molecule):
        """
        Initialises molecule to build conformational ensemble. 
        Most necassery molecule's attributes are initialised.
        
        Parameters
        ----------
        input_type (str): 'smiles' or 'molfile' required
        molecule str(str): if 'input_type' == 'smiles', treated as smiles; elif 'input_type' == 'molfile', treated as MOL file
        """        
        if input_type == 'smiles':
            self.smiles = molecule
            print(self.smiles)
            self.name = self.smiles.translate('/\*:()')
            self.mol = Chem.MolFromSmiles(self.smiles)

        elif input_type == 'molfile':
            self.mol = Chem.MolFromMolFile(molecule)
            self.smiles = Chem.MolToSmiles(self.mol)
            self.name = molecule.split('/')[-1].split('.')[0]
            self.mol = Chem.RemoveHs(self.mol)
    
        self.dihedrals = HiveMolecule.__find_torsions(self.mol)
        self.mol = Chem.AddHs(self.mol)           
        
        Chem.AllChem.EmbedMolecule(self.mol)
        
        # describe molecule as set of functional groups
        self.dof = len(self.dihedrals)     

        # attributes describing molecule during algorithm work
        self.xtb_call_count = 0
        self.alg_time = 0
        self.rmsd_time = 0
        
        # attributes describing chosen angles, energies and paths to conformations
        self.approved_dihedrals = []
        self.approved_xyzs = []
        self.E_list = []
        self.trajs = []
        self.rejected_dihedrals = []
        self.rejected_xyzs = []   
        self.visited = []
        print(self.mol, self.dof)
    def __get_xtbcc(self):
        """Get xtb call count"""
        return self.xtb_call_count
        
    def __set_xtbcc(self, value):
        """ Set xtb call count"""
        self.xtb_call_count = value
        
    def __str__(self):
        """#String representation of the molecule"""
        return f"SMILES: {self.smiles}\nDoF: {self.dof}\nXTB call count: {self.__get_xtbcc()}" 
    
    def get_dof(self):
        return self.dof

    
    @staticmethod    
    def __find_torsions(mol):
        """
        https://th.fhi-berlin.mpg.de/meetings/dft-workshop-2017/uploads/Meeting/Tutorial3_2017_manual.pdf
        https://github.com/adrianasupady/fafoom/blob/234d2a1c45f9ceb91de5f4d425de02c563cd9178/fafoom/deg_of_freedom.py
        Find the positions of rotatable bonds in the molecule.
        """
        # get only valuable dihedrals
        pattern_tor = Chem.MolFromSmarts("[*]~[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]~[*]")
        torsion = list(mol.GetSubstructMatches(pattern_tor))

        # remove duplicate torsion definitions from a list of atom ind. tuples.
        for_remove = []
        for x in reversed(range(len(torsion))):
            for y in reversed(range(x)):
                ix1, ix2 = itemgetter(1)(torsion[x]), itemgetter(2)(torsion[x])
                iy1, iy2 = itemgetter(1)(torsion[y]), itemgetter(2)(torsion[y])
                if (ix1 == iy1 and ix2 == iy2) or (ix1 == iy2 and ix2 == iy1):
                    for_remove.append(y)
        clean_list = [v for i, v in enumerate(torsion) if i not in set(for_remove)]
        return clean_list