import argparse
import os

import Cycle
from Molecule import HiveMolecule


def main():
    parser = argparse.ArgumentParser(description="Optimize a molecule using Hive.")
    parser.add_argument(
        "input",
        type=str,
        choices=['molfile', 'smiles'],
        help="Type of molecule input (.mol or smiles)",
    )
    parser.add_argument(
        "molecule",
        type=str,
        help="Molecule to be optimized (SMILES or .mol file path)",
    )
    parser.add_argument(
        "--n_confs",
        type=int,
        default=6,
        help="required number of conformations in enseble ",
    )      
    parser.add_argument(
        "--global_iterations",
        type=int,
        default=20,
        help="Number of ABC iterations of global conformation search",
    )
    parser.add_argument(
        "--local_iterations",
        type=int,
        default=20,
        help="Number of ABC iterations of local conformation search",
    )
    parser.add_argument(
        "--global_exists",
        type=str,
        default='',
        help="Provide the local search in respect to user's global conformation",
    )

    args = parser.parse_args()

    print(f"Molecule to be optimized: {args.molecule}")
    mol = HiveMolecule(args.input, args.molecule)
    ensemble = Cycle.full_cycle(mol, args.global_iterations, args.local_iterations, args.global_exists, args.n_confs)
    return ensemble

if __name__ == "__main__":
    main()