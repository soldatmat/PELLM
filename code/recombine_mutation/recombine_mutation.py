# Reproudction of the simulation of Directed Evolution: Recombining Mutations in Best Variants from https://www.pnas.org/doi/full/10.1073/pnas.1901979116.

import copy
import pandas

AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

MISSING_FITNESS = 0 # default fitness for missing variants
N_SAMPLES = 100
N_PARENTS = 3 benchmark

# Specific to data
MUTATION_POSITIONS = [0, 1, 2, 3] # [39, 40, 41, 54] in the sequence (indexed from 1)
WILD_TYPE_VARIANT = 'VDGV'
def LOAD_DATA():
    dfs = pandas.read_excel('../../data/GB1/elife-16965-supp1.xlsx')
    variants = [v for v in dfs.Variants]
    variant_fitness = dict(zip(dfs.Variants, dfs.Fitness))
    return variants, variant_fitness



def main():
    variants, variant_fitness = LOAD_DATA()

    
    
    return start_variants, final_variants, fitness_progressions  



if __name__ == "__main__":
    start_variants, final_variants, fitness_progressions = main()

    print(start_variants)
    print("___")
    print(final_variants)
    print(fitness_progressions)
