# ---- IMPORT MODULES

import copy
import random
import sys

import numpy as np

# ---- BEE CLASS

class FailedBee(Exception):
    pass

class Bee(object):
    """ Creates a bee object. """

    def __init__(self, lower, upper, fun, pickle, funcon=None):
        """
        Initialise random bee object

        Parameters:
        - lower (float): Object containing information about molecule.
        - upper (float): Object containing information about molecule.
        - fun (function): Object containing information about molecule.
        - pickle (float): Object containing information about molecule.

        Returns:
        - float: Minimum RMSD.
        """
        self._random(lower, upper)
        self.pickle = pickle
            
        self.value, self.fitness = fun(self.vector, self.pickle)
        self.counter = 0

    def _random(self, lower, upper):
        """Initialises a solution vector randomly."""

        self.vector = []
        for i in range(len(lower)):
            self.vector.append(lower[i] + random.random() * (upper[i] - lower[i]))

class BeeHive(object):


    def __init__(
        self,
        Type, 
        lower,
        upper,
        fun,
        numb_bees,
        max_itrs,
        max_trials,
        threshold,
        last_population,
        chosen_index,
        last_counter,
        allbees,
        allindexes,
        visited,
        pickle,
        glob_exists,
        seed=None
    ):
        """
        Parameters:
        - lower (list): lower bound of angles vector
        - upper (list): upper bound of angles vector
        - fun (function): function converting vector of molecule dihedral angles to Energy and Fitness Function
        - numb_bees (int): Number of bees, simultaneously flying above the PES
        - max_itrs (int): Number of algorithm iterations
        - max_trials (int): Maximum number of trials.
        - threshold (int): Minimum difference between the components of two vectors
        - last_population (list): Population of bees in the end of last algorithm work.
        - chosen_index (int): Index of the best bee of last algorithm work.
        - last_counter (list): Counter of failed improvement attempts during last algorithm work.
        - allbees (list): All vectors that lead to an unbroken molecule.
        - allindexes (list): Indexes of 'allbees' list which correspond to best conformations on every algorithm work
        - visited (list): All vectors found, including those leading to broken molecules
        - pickle (str): Path to pickle file containing required molecule parameteres.
        - glob_exists (bool): Determins type of search. "Global" if False and "Local" if True.
        - seed (int): variable for random.seed
        """

        assert len(upper) == len(lower)
        self.Type = Type
        self.lower = lower
        self.upper = upper
        self.evaluate = fun
        self.size = numb_bees
        self.dim = len(lower)
        self.max_itrs = max_itrs   
        self.max_trials = max_trials
        self.pickle = pickle
        self.threshold = threshold
        
        self.allbees = allbees
        self.allindexes = allindexes
        self.visited = visited
        self.last_population = last_population
        self.chosen_index = chosen_index
        self.last_counter = last_counter
        
        if seed is None:
            self.seed = random.randint(0, 1000)
        else:
            self.seed = seed
        print(f'seed: {self.seed}')
        random.seed(self.seed)

        self.population = []
        self.calculation_counter = 0
        self.failed_ladder_attempts = 0
        
        
        self.currentbees = []
        self.failed_attempts = 0
        self.best = sys.float_info.min
        
        if self.Type == 'Local':
            for i in range(len(self.population)):
                self.population[i] = self.last_population[i]            
            newbees = []
            if self.chosen_index is not None:
                while len(newbees) < 1:
                    newbee = Bee(lower, upper, fun, pickle)
                    print(newbee.vector)
                    if self.ladder_registration(newbee.vector, self.visited, threshold=self.threshold) is False or newbee.value == 0:
                        continue
                    else:
                        self.calculation_counter += 1
                        self.failed_ladder_attempts = 0
                        newbees.append(newbee)
                self.population[self.chosen_index] = newbees[0]
                self.population[self.chosen_index].counter = 0
            
                
                for i in range(len(self.population)):
                    if i != self.chosen_index:
                        self.population[i].counter = self.last_counter[i]
            
            else:
                for i in range(len(self.population)):
                        self.population[i].counter = 0              
            
            
                for i in self.population:
                    i.value, i.fitness = self.evaluate(i.vector, self.pickle)
                self.compute_local_probability()
                
        elif Type == 'Global':
            while len(self.population) < self.size:
                print('INITIAL HIVE')
                newbee = Bee(lower, upper, fun, pickle)
                print(np.around(newbee.vector, decimals=4))
                if self.ladder_registration(newbee.vector, self.visited, threshold=self.threshold) == False or newbee.value == 0:
                    continue
                else:
                    self.calculation_counter += 1
                    self.failed_ladder_attempts = 0
                    self.population.append(newbee)
                    self.visited.append(newbee.vector)
            
            self.currentbees.extend([bee for bee in self.population])
            self.find_best()

            self.compute_global_probability()


    def run(self):
        """Runs an Artificial Bee Colony (ABC) algorithm."""
        for itr in range(self.max_itrs):
            self.write_iteration(itr)

            self.write_current_data()
            self.write_send_employee(itr)
            for index in range(self.size):
                self.send_employee(index)
            self.write_current_data()
            self.write_end_send_employee(itr)

            self.write_send_onlooker(itr)
            self.send_onlookers()
            self.write_current_data()
            self.write_end_send_onlooker(itr)

            self.write_send_scout(itr)
            self.send_scout()
            self.write_current_data()
            self.write_end_send_scout(itr)
            
            self.find_best()
            self.write_best_on_iteration(itr)
            print('current vectors:', np.around([bee.vector for bee in self.population], decimals=4))
        self.counter = [bee.counter for bee in self.population]
        print('current vectors:', np.around([bee.vector for bee in self.population], decimals=4))
        print("BEST ON ITERATION:", np.around(self.solution, decimals=4))
        print( "self.best_index", self.best_index)
        self.allindexes.append(len(self.allbees) + self.currentindex)
        self.allbees.extend(self.currentbees)


    def find_best(self):
        """Finds current best bee candidate."""
        values = [bee.value for bee in self.population]
        fitnesses = [bee.fitness for bee in self.population]
        index_fit = fitnesses.index(max(fitnesses))
        print("values:", np.around(values, decimals=5))
        print("fitnesses:", np.around(fitnesses, decimals=5))
        print("best fit:", np.around(fitnesses[index_fit], decimals=5))

        if fitnesses[index_fit] > self.best:
            self.best = fitnesses[index_fit]
            self.solution = self.population[index_fit].vector
            self.best_index = index_fit

            current_vectors = [bee.vector for bee in self.currentbees]
            self.currentindex = current_vectors.index(self.solution)

    def compute_global_probability(self):
        """
        Computing probability of every bee in population of being chosen. 
        Probability calculation is based on current population and depends on 
        best and worst conformations. Used in global minumum search
        
        Returns:
        - list: probabilities of being chosen by onlooker bee
        
        """
        values = [bee.value for bee in self.population]
        best, worst = abs(min(values)), abs(max(values))
        self.probas = [0.2 + 0.8 * (abs(v) - worst) / (best - worst) for v in values]

        return [sum(self.probas[: i + 1]) for i in range(self.size)] 
    
    
    def compute_local_probability(self):
        """
        Computing probability of every bee in population of being chosen. 
        Probability calculation depends on global conformation's energy. 
        Used in local minumum search
        
        Returns:
        - list: probabilities of being chosen by onlooker bee
        """
        values = [bee.fitness for bee in self.population]
        max_values = max(values)

        self.probas = [0.9 * v / max_values + 0.1 for v in values]
        print(
            "Computing probability of being chosen. Depends on type of search",
            f"based on fitness: {values}",
            f"Probabilities {np.around(self.probas, decimals = 4)}",
            sep="\n",
        )

        return [sum(self.probas[: i + 1]) for i in range(self.size)]
            
    def send_employee(self, index):
        """Send employee bees

        Parameters:
            index (int): index of upgraded bee in population
        """  
        zombee = copy.deepcopy(self.population[index])
        zombee.value, zombee.fitness = 0, 0
        while zombee.value == 0 and zombee.fitness == 0:
            zombee = copy.deepcopy(self.population[index])
            d = random.randint(0, self.dim - 1)
            bee_ix = index
            while bee_ix == index:
                bee_ix = random.randint(0, self.size - 1)

            zombee.vector[d] = self.mutate(d, index, bee_ix)
            zombee.vector = self.check(zombee.vector)
            print(
                f"\nupgraded bee: {index}            another bee: {bee_ix}        dimension to change:{d}",
                f"upgraded bee: {np.around(self.population[index].vector, decimals =4)}",
                f"another bee: {np.around(self.population[bee_ix].vector, decimals =4)}",
                f"candidate bee: {np.around(zombee.vector, decimals =4)}\n",
                sep="\n",
            )
            if (
                self.ladder_registration(
                    zombee.vector, self.visited, threshold=self.threshold
                )
                is False
            ):
                continue
            else:
                
                self.failed_ladder_attempts = 0
                self.visited.append(copy.deepcopy(zombee.vector))
                zombee.value, zombee.fitness = self.evaluate(zombee.vector, self.pickle)
                if zombee.value == 0:
                    print("Molecule crashes", end="\n")
                    continue
                else:
                    self.calculation_counter += 1
        self.currentbees.append(zombee)

        if zombee.fitness > self.population[index].fitness:
            self.population[index] = copy.deepcopy(zombee)
            self.population[index].counter = 0
            print(f"bee {index} upgraded", end="\n-------\n")
        else:
            self.population[index].counter += 1
            print(f"bee {index} stayed same", end="\n-------\n\n")

    def send_onlookers(self):
        """Send onlookers phase """
        numb_onlookers = 0
        beta = 0
        while numb_onlookers < self.size:
            phi = random.random()
            beta += phi * max(self.probas)
            beta %= sum(self.probas)
            index = self.select(beta)
            print(
                f"phi: {np.around(phi, decimals = 4)}",
                f"beta: {np.around(beta, decimals = 4)}",
                f"selected bee: {index}",
                f"counter of chosen bee:{self.population[index].counter}",
                sep="\n",
            )
            self.send_employee(index)
            numb_onlookers += 1


    def select(self, beta):
        """selection of bee to improve based on improvement probability

        Parameters:
            beta (float): Fitness proportionate selection parameter
            
        Returns:
            index (int): index of bee to be upgraded by onlooker bee
        """  
        probas = self.compute_probability()
        print(f"probas:  {np.around(probas, decimals = 4)}")

        for index in range(self.size):
            if beta < probas[index]:
                return index
        return index



    def send_scout(self):
        """Send scout bees"""
        bees_for_change = []
        for i in range(len(self.population)):
            if self.population[i].counter > self.max_trials:
                bees_for_change.append(i)
        if len(bees_for_change) > 0:
            newbees = []
            while len(newbees) < len(bees_for_change):
                newbee = Bee(self.lower, self.upper, self.evaluate, self.pickle)
                if (
                        self.ladder_registration(
                            newbee.vector, self.visited, threshold=self.threshold
                        )
                        is False
                    ):
                    continue
                else:
                    self.failed_ladder_attempts = 0
                    self.visited.append(copy.deepcopy(newbee.vector))
                    if newbee.value == 0:
                         continue
                    else:
                        self.calculation_counter += 1
                        newbees.append(newbee)
                        self.currentbees.append(newbee)

            for index in range(len(bees_for_change)):
                self.population[bees_for_change[index]] = newbees[index]
                self.send_employee(bees_for_change[index])


    def mutate(self, dim, current_bee, other_bee):
        """
        mutation of one component of the vector to explore the vicinity of the PES

        Parameters:
            dim (int): Fitness proportionate selection parameter
            current_bee (int): index of current bee
            other_bee (int): index of another bee to cross-over
            
        Returns:
        list: new mutated vector
        """
        return self.population[current_bee].vector[dim] + (
            random.random() - 0.5
        ) * 2 * (
            self.population[current_bee].vector[dim]
            - self.population[other_bee].vector[dim]
        )

    
    def compute_probability(self):
        """choosing probability calculating function depending on type of search"""
        if self.Type == 'Global':
            return self.compute_global_probability()
        elif self.Type == 'Local':
            return self.compute_local_probability()
        
    def check(self, vector):
        """
        Check and correct the bounds of solution vector

        Parameters:
        - vector (list): vector to check and correct bounds
        
        Returns:
        - vector (list): vector with corrected bounds
        """

        for i in range(self.dim):

            if vector[i] < self.lower[i]:
                vector[i] = (np.absolute(vector[i]) - 180) % 360
            elif vector[i] > self.upper[i]:
                vector[i] = -(360 - vector[i]) % -360
        return vector


    def ladder_registration(self, candidate, visited_list, threshold):
        """
        Check the diversity condition of new conformation

        Parameters:
        - candidate (list): vector to pass diversity condition
        - visited_list (list): list of all vectors checked before the current vector
        - threshold (float): Minimum difference between the components of candidate and every vector in visited_list
        
        Returns:
        - bool: returns whether the diversity condition is satisfied
        """
        for i in range(self.dim):
            step = []
            for visited in visited_list:
                if np.abs(candidate[i] - visited[i]) < threshold:
                    step.append(visited)
            visited_list = step
            if len(visited_list) == 0:
                return True
            else:
                continue
        print(
            f"{np.around(candidate, decimals=4)} crashed with {np.around(visited_list[0], decimals=4)}"
        )
        print("Space have been discovered before", end="\n\n")
        self.failed_ladder_attempts += 1
        if self.failed_ladder_attempts == 30:
            raise FailedBee
        return False

    def write_current_data(self):
        print('Energies:', np.around([bee.value for bee in self.population], decimals=5))
        print('FF:', np.around([bee.fitness for bee in self.population], decimals=5))
        print('Probabilities', np.around(self.compute_probability(), decimals=5))

    def write_iteration(self, itr):
        text = f"ITERATION {itr}"
        text_centered = text.center(90, "_")
        print(text_centered)

    def write_send_employee(self, itr):
        text = f"{itr} SEND_EMPLOYEE {itr}"
        text_centered = text.center(90, "-")
        print(text_centered)

    def write_end_send_employee(self, itr):
        text = f"{itr} END_SEND_EMPLOYEE {itr}"
        text_centered = text.center(90, "-")
        print(text_centered)

    def write_send_onlooker(self, itr):
        text = f"{itr} SEND_ONLOOKERS {itr}"
        text_centered = text.center(90, "-")
        print(text_centered)

    def write_end_send_onlooker(self, itr):
        text = f"{itr} END_SEND_ONLOOKERS {itr}"
        text_centered = text.center(90, "-")
        print(text_centered)

    def write_send_scout(self, itr):
        text = f"{itr} SEND_SCOUT {itr}"
        text_centered = text.center(90, "-")
        print(text_centered)

    def write_end_send_scout(self, itr):
        text = f"{itr} END_SEND_SCOUT {itr}"
        text_centered = text.center(90, "-")
        print(text_centered)

    def write_best_on_iteration(self, itr):
        text = f"{itr} best on iteration   {np.around(self.best,  decimals=4)}   {np.around(self.solution, decimals=4)} {itr}"
        text_centered = text.center(90, "-")
        print(text_centered)

