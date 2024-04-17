using DESilico
using CSV
using DataFrames
using PyCall
using FileIO

torch = pyimport("torch")

pushfirst!(pyimport("sys")."path", joinpath(@__DIR__))
nn_model = pyimport("two_layer_perceptron")

include("types.jl")
include("utils.jl")
include("embedding_nn.jl")
include("dict_embedding_extractor.jl")
include("top_k_predicted.jl")
include("prediction_distance_maximizer.jl")
include("no_mutagenesis.jl")
include("distance_maximizer.jl")
include("cumulative_select.jl")

GC.gc() # For re-runs, GPU allocs by Python sometimes do not get freed automatically

# ___ Data specific parameters ___
# GB1
data_path = joinpath(@__DIR__, "..", "..", "data", "GB1")
xlxs_file = "elife-16965-supp1.xlsx"
wt_string = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"  # ['V', 'D', 'G', 'V']
mutation_positions = [39, 40, 41, 54]
missing_fitness_value = 0.0

# PhoQ
#= data_path = joinpath(@__DIR__, "..", "..", "data", "PhoQ")
xlsx_file = "PhoQ.xlsx"
wt_string = "MKKLLRLFFPLSLRVRFLLATAAVVLVLSLAYGMVALIGYSVSFDKTTFRLLRGESNLFYTLAKWENNKLHVELPENIDKQSPTMTLIYDENGQLLWAQRDVPWLMKMIQPDWLKSNGFHEIEADVNDTSLLLSGDHSIQQQLQEVREDDDDAEMTHSVAVNVYPATSRMPKLTIVVVDTIPVELKSSYMVWSWFIYVLSANLLLVIPLLWVAAWWSLRPIEALAKEVRELEEHNRELLNPATTRELTSLVRNLNRLLKSERERYDKYRTTLTDLTHSLKTPLAVLQSTLRSLRSEKMSVSDAEPVMLEQISRISQQIGYYLHRASMRGGTLLSRELHPVAPLLDNLTSALNKVYQRKGVNISLDISPEISFVGEQNDFVEVMGNVLDNACKYCLEFVEISARQTDEHLYIVVEDDGPGIPLSKREVIFDRGQRVDTLRPGQGVGLAVAREITEQYEGKIVAGESMLGGARMEVIFGRQHSAPKDE"
mutation_positions = [284, 285, 288, 289]
missing_fitness_value = 0.0 =#

alphabet = DESilico.alphabet.protein
wt_sequence = collect(wt_string)
sequences = _get_sequences(data_path, "esm-1b_variants.csv", wt_string, mutation_positions)
sequences_complete = _get_sequences(data_path, "esm-1b_variants_complete.csv", wt_string, mutation_positions)
fitness = _get_fitness(data_path, "esm-1b_fitness_norm.csv")
sequence_embeddings = _get_sequence_emebeddings(data_path, "esm-1b_embedding_complete.csv")

sequence_embeddings_a = CSV.read(joinpath(data_path, "esm-1b_embedding_complete.csv"), DataFrame)
sequence_embeddings_a = Matrix{Float64}(sequence_embeddings_a)
sequence_embeddings_a = transpose(sequence_embeddings_a)

# ___ Screening ___
screening = DESilico.DictScreening(Dict(sequences .=> fitness), missing_fitness_value)

# ___ Passive Sampling ___
wt_variant = Variant(wt_sequence, screening(wt_sequence))
sequence_space = SequenceSpace{Vector{Variant}}([wt_variant])
distance_maximizer = DistanceMaximizer(sequences_complete, sequence_embeddings_a)
cumulative_select = CumulativeSelect(sequence_space.population)
de!(
    sequence_space;
    screening=screening,
    selection_strategy=cumulative_select,
    mutagenesis=distance_maximizer,
    n_iterations=384,
)

# ___ Active Learning ___
#sequence_space.population = Vector{Variant}([])

#mutagenesis = NoMutagenesis()

embedding_extractor = DictEmbeddingExtractor(Dict(sequences_complete .=> sequence_embeddings))
fp_model = nn_model.TwoLayerPerceptron(torch.nn.Sigmoid(), embedding_extractor.embedding_size)
fitness_predictor = EmbeddingNN(embedding_extractor, fp_model)
#selection_strategy = TopKPredicted(fitness_predictor, sequences_complete; k=1, repeat=false)
#selection_strategy = PredictionDistanceMaximizer(fitness_predictor, sequences_complete; screened=sequence_space.variants, k=1, repeat=false)

#= de!(
    sequence_space;
    screening,
    selection_strategy,
    mutagenesis,
    n_iterations=360,
) =#
train!(fitness_predictor, sequence_space.variants)

println(sequence_space.top_variant)
length(sequence_space.variants)
map(v -> extract_residues(v.sequence, mutation_positions), sequence_space.variants)

prediction = fitness_predictor(sequences_complete)
pairs = map(i -> (sequences_complete[i], prediction[i]), eachindex(sequences_complete))
sort!(pairs, by=x -> x[2], rev=true)
println(sum(map(pair->screening(pair[1]), pairs[1:100]))/100)
println(maximum(map(pair->screening(pair[1]), pairs[1:100])))

# ___ Save Results ___
history = reconstruct_history(sequence_space.variants)
top_sequence = map(v -> v.sequence, history)
top_variant = map(sequence -> extract_residues(sequence, mutation_positions), top_sequence)
top_fitness = map(v -> v.fitness, history)
save_path = joinpath(@__DIR__, "data", "mlde_input_distmax_01")
save(
    joinpath(save_path, "de.jld2"),
    "sequence_space", sequence_space,
    "selection_strategy", selection_strategy,
    "mutagenesis", mutagenesis,
    "history", history,
)
save(
    joinpath(save_path, "history.jld2"),
    "top_sequence", top_sequence,
    "top_variant", top_variant,
    "top_fitness", top_fitness,
)
torch.save(fp_model.state_dict(), joinpath(save_path, "TwoLayerPreceptron_state_dict.pt"))
