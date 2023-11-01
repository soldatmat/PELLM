# Reproudction of the simulation of Directed Evolution: Recombining Mutations in Best Variants from https://www.pnas.org/doi/full/10.1073/pnas.1901979116.

import copy
import pandas
import random

AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

MIN_FITNESS = 0
MISSING_FITNESS = 0 # default fitness for missing variants
N_SAMPLES = 81
N_PARENTS = 3
N_ITER = 7

# Specific to data
MUTATION_POSITIONS = [0, 1, 2, 3] # [39, 40, 41, 54] in the sequence (indexed from 1)
WILD_TYPE_VARIANT = 'VDGV'
def LOAD_DATA():
    dfs = pandas.read_excel('../../data/GB1/elife-16965-supp1.xlsx')
    variants = [v for v in dfs.Variants]
    variant_fitness = dict(zip(dfs.Variants, dfs.Fitness))
    return variants, variant_fitness



def recombine_mutation(recombinatorial_lib, mutation_positions, fitness_dict, n_parents=N_PARENTS, n_iter=N_ITER):
    best_variant = None
    fitness_progression = [MIN_FITNESS-1]

    lib = recombinatorial_lib
    for i in range(n_iter):
        # get top [n_parents] variants from lib
        evaluated_lib = []
        for variant in lib:
            fitness = fitness_dict[variant]

            # update fitness progress
            if fitness > fitness_progression[-1]:
                best_variant = variant
                fitness_progression.append(fitness)
            else:
                fitness_progression.append(fitness_progression[-1])

            evaluated_lib.append((variant, fitness))
        evaluated_lib.sort(key=lambda tup: tup[1], reverse=True)
        parents = evaluated_lib[:n_parents]

        if i == n_iter - 1:
            break

        # recombine parents
        lib = [parents[0][0]]
        for p in mutation_positions:
            new_variants = []
            for variant in lib:
                for parent in parents:
                    pos = mutation_positions[p]
                    if variant[pos] == parent[0][pos]:
                        continue
                    mutated_variant = variant[:pos] + parent[0][pos] + variant[pos+1:]
                    new_variants.append(mutated_variant)
            lib = lib + new_variants

    return best_variant, fitness_progression[1:]


def main():
    variants, variant_fitness = LOAD_DATA()
    combinatorial_lib = list(variant_fitness) # gets keys of dict as list
    recombinatorial_lib = random.sample(combinatorial_lib, N_SAMPLES)
    best_variant, fitness_progression = recombine_mutation(recombinatorial_lib, MUTATION_POSITIONS, variant_fitness)
    
    return best_variant, fitness_progression  



if __name__ == "__main__":
    best_variant, fitness_progression = main()

    print(fitness_progression)
    print(best_variant)
