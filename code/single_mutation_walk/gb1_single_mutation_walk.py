# Reproudction of the simulation of Directed Evolution: Single Mutation Walk from https://www.pnas.org/doi/full/10.1073/pnas.1901979116.

import copy
import pandas

AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

MISSING_FITNESS = 0. # default fitness for missing variants
START_OPTIONS = ['wt', 'all'] # 'wt' - start from the wild type variant, 'all' - run single-mutation walk from each variant

# Specific to data
MUTATION_POSITIONS = [0, 1, 2, 3] # [39, 40, 41, 54] in the sequence (indexed from 1)
WILD_TYPE_VARIANT = 'VDGV'
def LOAD_DATA():
    dfs = pandas.read_excel('../../data/GB1/elife-16965-supp1.xlsx')
    variants = [v for v in dfs.Variants]
    variant_fitness = dict(zip(dfs.Variants, dfs.Fitness))
    return variants, variant_fitness



def single_mutation_walk(start_variant, mutation_positions, fitness_dict):
    variant = start_variant
    unresolved_positions = copy.deepcopy(mutation_positions)
    fitness_progression = [fitness_dict[variant]]
    while unresolved_positions:
        # fix an amino acid at an unresolved position which gives the highest fitness
        best_p = mutation_positions[0]
        best_aa = variant[mutation_positions[0]]
        for p in range(len(unresolved_positions)):
            pos = unresolved_positions[p]
            for aa in AMINO_ACIDS:
                if aa == variant[pos]:
                    continue
                mutated_variant = variant[:pos] + aa + variant[pos+1:]
                fitness = fitness_dict.get(mutated_variant, MISSING_FITNESS)
                if fitness > fitness_progression[-1]:
                    best_p = p
                    best_aa = aa
                    fitness_progression.append(fitness)
                else:
                    fitness_progression.append(fitness_progression[-1])
                
        
        if best_p is not None:
            best_pos = unresolved_positions[best_p]
            variant = variant[:best_pos] + best_aa + variant[best_pos+1:]
        unresolved_positions.pop(best_p)

    return variant, fitness_progression


def main(start=START_OPTIONS[0]):
    variants, variant_fitness = LOAD_DATA()

    if start == START_OPTIONS[0]:
        start_variants = [WILD_TYPE_VARIANT]
    elif start == START_OPTIONS[1]:
        start_variants = variants # TODO start only from reported variants
    else:
        raise(ValueError("start must be one of the following: " + str(START_OPTIONS)))

    final_variants = []
    fitness_progressions = []
    for variant in start_variants:
        # single-muation walk starting from the variant
        mutated_variant, fitness_progression = single_mutation_walk(variant, MUTATION_POSITIONS, variant_fitness)
        final_variants.append(mutated_variant)
        fitness_progressions.append(fitness_progression)
    
    return start_variants, final_variants, fitness_progressions  



if __name__ == "__main__":
    start_variants, final_variants, fitness_progressions = main()

    print(start_variants)
    print("___")
    print(final_variants)
    print(fitness_progressions)
