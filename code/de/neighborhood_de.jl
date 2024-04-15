using DESilico
using CSV
using FileIO
using PyCall

torch = pyimport("torch")

pushfirst!(pyimport("sys")."path", joinpath(@__DIR__))
nn_model = pyimport("two_layer_perceptron")

include("types.jl")
include("utils.jl")
include("construct_neighborhoods.jl")
include("distance_maximizer.jl")
include("library_select.jl")
include("cumulative_select.jl")
include("repeating_library_select.jl")
include("neighborhood_search.jl")
include("centrality_maximizer.jl")
include("training_screening.jl")
include("embedding_nn.jl")
include("dict_embedding_extractor.jl")

GC.gc() # For re-runs, GPU allocs by Python sometimes do not get freed automatically

# ___ Data specific parameters ___
# GB1
data_path = joinpath(@__DIR__, "..", "..", "data", "GB1")
wt_string = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"  # ['V', 'D', 'G', 'V']
mutation_positions = [39, 40, 41, 54]
missing_fitness_value = 0.0
neighborhoods_filename = "gb1_esm1b_euclidean.jld2"

# PhoQ
#= data_path = joinpath(@__DIR__, "..", "..", "data", "PhoQ")
wt_string = "MKKLLRLFFPLSLRVRFLLATAAVVLVLSLAYGMVALIGYSVSFDKTTFRLLRGESNLFYTLAKWENNKLHVELPENIDKQSPTMTLIYDENGQLLWAQRDVPWLMKMIQPDWLKSNGFHEIEADVNDTSLLLSGDHSIQQQLQEVREDDDDAEMTHSVAVNVYPATSRMPKLTIVVVDTIPVELKSSYMVWSWFIYVLSANLLLVIPLLWVAAWWSLRPIEALAKEVRELEEHNRELLNPATTRELTSLVRNLNRLLKSERERYDKYRTTLTDLTHSLKTPLAVLQSTLRSLRSEKMSVSDAEPVMLEQISRISQQIGYYLHRASMRGGTLLSRELHPVAPLLDNLTSALNKVYQRKGVNISLDISPEISFVGEQNDFVEVMGNVLDNACKYCLEFVEISARQTDEHLYIVVEDDGPGIPLSKREVIFDRGQRVDTLRPGQGVGLAVAREITEQYEGKIVAGESMLGGARMEVIFGRQHSAPKDE"
mutation_positions = [284, 285, 288, 289]
missing_fitness_value = 0.0
neighborhoods_filename = "phoq_esm1b_euclidean.jld2" =#

# ___ Load data ___
wt_sequence = collect(wt_string)
sequences = _get_sequences(data_path, "esm-1b_variants.csv", wt_string, mutation_positions)
sequences_complete = _get_sequences(data_path, "esm-1b_variants_complete.csv", wt_string, mutation_positions)
fitness = _get_fitness(data_path, "esm-1b_fitness_norm.csv")
sequence_embeddings = _get_sequence_emebeddings(data_path, "esm-1b_embedding_complete.csv")

# ___ Neighborhoods ___
sequence_embeddings_a = CSV.read(joinpath(data_path, "esm-1b_embedding_complete.csv"), DataFrame)
sequence_embeddings_a = Matrix{Float64}(sequence_embeddings_a)
sequence_embeddings_a = transpose(sequence_embeddings_a)
# neighborhoods = _construct_neighborhoods(sequence_embeddings_n)
neighborhoods = load(joinpath(@__DIR__, "data", "neighborhoods", neighborhoods_filename))["neighborhoods"]

# ___ Fitness Predictor ___
embedding_extractor = DictEmbeddingExtractor(Dict(sequences_complete .=> sequence_embeddings))
fp_model = nn_model.TwoLayerPerceptron(torch.nn.Sigmoid(), embedding_extractor.embedding_size)
fitness_predictor = EmbeddingNN(embedding_extractor, fp_model)

# ___ Screening ___
screening = DESilico.DictScreening(Dict(sequences .=> fitness), missing_fitness_value)
training_screening = TrainingScreening(screening, (variants) -> train!(fitness_predictor, variants))

# ___ Change starting variant ___
#starting_mutant = "HECA" # bad: "DAKQ", good: "SRCG" "HECA"
#wt_sequence = collect(wt_string[1:mutation_positions[1]-1] * starting_mutant[1] * wt_string[mutation_positions[1]+1:mutation_positions[2]-1] * starting_mutant[2] * wt_string[mutation_positions[2]+1:mutation_positions[3]-1] * starting_mutant[3] * wt_string[mutation_positions[3]+1:mutation_positions[4]-1] * starting_mutant[4] * wt_string[mutation_positions[4]+1:end])

# ___ Run de! ___
wt_variant = Variant(wt_sequence, screening(wt_sequence))
sequence_space = SequenceSpace{Vector{Variant}}([wt_variant])

distance_maximizer = DistanceMaximizer(sequences_complete, sequence_embeddings_a)
cumulative_select = CumulativeSelect(sequence_space.population)
de!(
    sequence_space;
    screening=training_screening,
    selection_strategy=cumulative_select,
    mutagenesis=distance_maximizer,
    n_iterations=9,
)
starting_variants = sequence_space.variants

#starting_variants = map(i -> Variant(sequences_complete[i], screening(sequences_complete[i])), get_top_centrality(neighborhoods, 10))

knn = 16
#= neighborhood_search = NeighborhoodSearch(
    sequences_complete,
    neighborhoods[1:knn, :];
    repeat=false,
    screened=map(variant -> variant.sequence, collect(starting_variants)),
    #n=1,
) =#
neighborhood_search = NeighborhoodSearchWithPredictor(
    sequences_complete,
    neighborhoods[1:knn, :];
    repeat=false,
    screened=map(variant -> variant.sequence, collect(starting_variants)),
    predictor=fitness_predictor,
)

library_select = LibrarySelect(1, starting_variants)
#library_select = LibrarySelect(1, Vector{Variant}([]))
#library_select = SamplingLibrarySelect(1, starting_variants, distance_maximizer, screening, sequence_space)
#library_select = RepeatingLibrarySelect()

parent_sequence = library_select()[1]
sequence_space = SequenceSpace{Vector{Variant}}([Variant(parent_sequence, screening(parent_sequence))])
DESilico.push_variants!(sequence_space, collect(library_select.library))
de!(
    sequence_space;
    screening=training_screening,
    selection_strategy=library_select,
    mutagenesis=neighborhood_search,
    n_iterations=nothing,
    screening_budget=384,
)

# 23 (239) for knn=16 to get 1.0 fitness, (RepeatingLibrarySelect + n=1 in n.s.) => (230)
# 17 (280) for knn=24 to get 1.0 fitness
# 22 (430) for knn=32 to get 1.0 fitness
println(sequence_space.top_variant)
length(Set(sequence_space.variants))
