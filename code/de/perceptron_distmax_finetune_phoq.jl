using DESilico
using CSV
using DataFrames
using PyCall
using FileIO

torch = pyimport("torch")

pushfirst!(pyimport("sys")."path", joinpath(@__DIR__))
nn_model = pyimport("two_layer_perceptron")

include("types.jl")
include("llm.jl")
include("utils.jl")
include("embedding_nn.jl")
include("esm1b.jl")
include("llm_embedding_extractor.jl")
include("dict_embedding_extractor.jl")
include("top_k_predicted.jl")
include("llm_sampler.jl")
include("distance_maximizer.jl")
include("cumulative_select.jl")
include("prediction_distance_maximizer.jl")
include("no_mutagenesis.jl")

GC.gc() # For re-runs, GPU allocs by Python sometimes do not get freed automatically

# ___ Data specific parameters ___
# GB1
#= data_path = joinpath(@__DIR__, "..", "..", "data", "GB1")
xlxs_file = "elife-16965-supp1.xlsx"
wt_string = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"  # ['V', 'D', 'G', 'V']
mutation_positions = [39, 40, 41, 54]
missing_fitness_value = 0.0 =#

# PhoQ
data_path = joinpath(@__DIR__, "..", "..", "data", "PhoQ")
xlsx_file = "PhoQ.xlsx"
wt_string = "MKKLLRLFFPLSLRVRFLLATAAVVLVLSLAYGMVALIGYSVSFDKTTFRLLRGESNLFYTLAKWENNKLHVELPENIDKQSPTMTLIYDENGQLLWAQRDVPWLMKMIQPDWLKSNGFHEIEADVNDTSLLLSGDHSIQQQLQEVREDDDDAEMTHSVAVNVYPATSRMPKLTIVVVDTIPVELKSSYMVWSWFIYVLSANLLLVIPLLWVAAWWSLRPIEALAKEVRELEEHNRELLNPATTRELTSLVRNLNRLLKSERERYDKYRTTLTDLTHSLKTPLAVLQSTLRSLRSEKMSVSDAEPVMLEQISRISQQIGYYLHRASMRGGTLLSRELHPVAPLLDNLTSALNKVYQRKGVNISLDISPEISFVGEQNDFVEVMGNVLDNACKYCLEFVEISARQTDEHLYIVVEDDGPGIPLSKREVIFDRGQRVDTLRPGQGVGLAVAREITEQYEGKIVAGESMLGGARMEVIFGRQHSAPKDE"
mutation_positions = [284, 285, 288, 289]
missing_fitness_value = 0.0

alphabet = DESilico.alphabet.protein
wt_sequence = collect(wt_string)
sequences = _get_sequences(data_path, "esm-1b_variants.csv", wt_string, mutation_positions)
sequences_complete = _get_sequences(data_path, "esm-1b_variants_complete.csv", wt_string, mutation_positions)
fitness = _get_fitness(data_path, "esm-1b_fitness_norm.csv")
sequence_embeddings = _get_sequence_emebeddings(data_path, "esm-1b_embedding_complete.csv")

sequence_embeddings_a = CSV.read(joinpath(data_path, "esm-1b_embedding_complete.csv"), DataFrame)
sequence_embeddings_a = Matrix{Float64}(sequence_embeddings_a)
sequence_embeddings_a = transpose(sequence_embeddings_a)

# ___ LLM ___
llm = ESM1b()
mask_token = llm.alphabet.mask_idx
mask_string = llm.alphabet.all_toks[llm.alphabet.mask_idx+1] # +1 for julia vs Python indexing

# ___ Screening ___
#screening = DESilico.DictScreening(joinpath(data_path, xlxs_file), missing_fitness_value)
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
    n_iterations=23,
)

# ___ Active Learning ___
embedding_extractor = LLMEmbeddingExtractor(llm; batch_size=100, return_tensor=true)
#embedding_extractor = DictEmbeddingExtractor(Dict(sequences_complete .=> sequence_embeddings))
fp_model = nn_model.TwoLayerPerceptron(torch.nn.Sigmoid(), embedding_extractor.embedding_size)
fitness_predictor = EmbeddingNN(embedding_extractor, fp_model)

#selection_strategy = TopKPredicted(fitness_predictor, length(variants[1]), alphabet; k=8000)
#selection_strategy = TopKPredicted(fitness_predictor, sequences_complete; k=8000) # AFP-DE k=8000 (! + 1000 other sequences)
selection_strategy = PredictionDistanceMaximizer(fitness_predictor, sequences_complete; screened=sequence_space.variants, k=24, repeat=false, train_callback=train_llm)
function train_llm()
    println("Finetuning LLM ...")
    prediction = fitness_predictor(selection_strategy.sequences)
    predicted_variants = map((s, f) -> Variant(s, f), selection_strategy.sequences, prediction)
    selected_sequences = select_top_k!(selection_strategy, predicted_variants)
    train!(llm, selected_sequences; mask_positions=mutation_posisitions)
end

#alphabet_extractor = LLMSampler(llm; sampling_sequence=wt_sequence, alphabet, k=3) # k=3 from AFP-DE
#mutagenesis = DESilico.Recombination(alphabet_extractor; mutation_positions, n=24) # n=24 from AFP-DE
mutagenesis=NoMutagenesis()

sequence_space.population = selection_strategy(sequence_space.variants)
de!(
    sequence_space;
    screening,
    selection_strategy,
    mutagenesis,
    n_iterations=15,
)
println(sequence_space.top_variant)

# ___ Save Results ___
history = reconstruct_history(sequence_space.variants)
top_sequence = map(v -> v.sequence, history)
top_variant = map(sequence -> extract_residues(sequence, mutation_positions), top_sequence)
top_fitness = map(v -> v.fitness, history)
save_path = joinpath(@__DIR__, "data", "perceptron_distmax_finetune_phoq_01")
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

# ___ Plot results ___
#using Plots
#histogram(map(v->v.fitness, [v for v in sequence_space.variants]), bins=range(0,1,length=20))

