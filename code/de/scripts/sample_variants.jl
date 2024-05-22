using CSV
using FileIO
using StatsBase
using Plots

include("../utils.jl")

# ___ Data specific parameters ___
# GB1
data_path = joinpath(@__DIR__, "..", "..", "..", "data", "GB1")
wt_string = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"  # ['V', 'D', 'G', 'V']
mutation_positions = [39, 40, 41, 54]
missing_fitness_value = 0.0
neighborhoods_filename = "gb1_esm1b_euclidean.jld2"

# PhoQ
#= data_path = joinpath(@__DIR__, "..", "..", "..", "data", "PhoQ")
wt_string = "MKKLLRLFFPLSLRVRFLLATAAVVLVLSLAYGMVALIGYSVSFDKTTFRLLRGESNLFYTLAKWENNKLHVELPENIDKQSPTMTLIYDENGQLLWAQRDVPWLMKMIQPDWLKSNGFHEIEADVNDTSLLLSGDHSIQQQLQEVREDDDDAEMTHSVAVNVYPATSRMPKLTIVVVDTIPVELKSSYMVWSWFIYVLSANLLLVIPLLWVAAWWSLRPIEALAKEVRELEEHNRELLNPATTRELTSLVRNLNRLLKSERERYDKYRTTLTDLTHSLKTPLAVLQSTLRSLRSEKMSVSDAEPVMLEQISRISQQIGYYLHRASMRGGTLLSRELHPVAPLLDNLTSALNKVYQRKGVNISLDISPEISFVGEQNDFVEVMGNVLDNACKYCLEFVEISARQTDEHLYIVVEDDGPGIPLSKREVIFDRGQRVDTLRPGQGVGLAVAREITEQYEGKIVAGESMLGGARMEVIFGRQHSAPKDE"
mutation_positions = [284, 285, 288, 289]
missing_fitness_value = 0.0
neighborhoods_filename = "phoq_esm1b_euclidean.jld2" =#

wt_sequence = collect(wt_string)
variants = _get_variants(data_path, "esm-1b_variants.csv")
fitness = _get_fitness(data_path, "esm-1b_fitness_norm.csv")

pairs = map(i -> (variants[i], fitness[i]), eachindex(variants))
selection = sample(pairs, 1000, replace=false)

histogram(fitness)
histogram(fitness[1:1000])
histogram(map(pair -> pair[2], selection))

save(
    joinpath(data_path, "sample_1000_02.jld2"),
    "variants", map(pair -> pair[1], selection),
    "fitness", map(pair -> pair[2], selection),
)



variants_complete = _get_variants(data_path, "esm-1b_variants_complete.csv")
sequences_complete = _construct_sequences(variants_complete, wt_string, mutation_positions)
sequences = _construct_sequences(variants, wt_string, mutation_positions)
screening = DESilico.DictScreening(Dict(sequences .=> fitness), missing_fitness_value)

f = map(s -> screening(s), sequences_complete[1:5000])
histogram(f)
