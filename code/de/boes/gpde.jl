using CSV
using DataFrames
using FileIO
using DESilico
using BOSS
using OptimizationPRIMA

include("../utils.jl")
include("lib/embedding_gp.jl")
include("lib/de_acq_maximizer.jl")

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

# ___ Load data ___
wt_sequence = collect(wt_string)
wt_variant = map(pos -> wt_sequence[pos], mutation_positions)
alphabet = collect(DESilico.alphabet.protein)
domain_encoder = Dict(alphabet .=> eachindex(alphabet))
_encode_domain_float(variant::AbstractVector{Char}) = map(symbol -> Float64.(domain_encoder[symbol]), variant)
_encode_domain(variant::AbstractVector{Char}) = map(symbol -> domain_encoder[symbol], variant)
domain_decoder = Dict(eachindex(alphabet) .=> alphabet)
_decode_domain(coords::AbstractVector{Int}) = map(coord -> domain_decoder[coord], coords)

variants_complete = _get_variants(data_path, "esm-1b_variants_complete.csv")
variant_coords = map(variant -> _encode_domain(variant), variants_complete)
sequences_complete = _construct_sequences(variants_complete, wt_string, mutation_positions)
sequences = _get_sequences(data_path, "esm-1b_variants.csv", wt_string, mutation_positions)
fitness = _get_fitness(data_path, "esm-1b_fitness_norm.csv")
sequence_embeddings = _get_sequence_emebeddings(data_path, "esm-1b_embedding_complete.csv")

# ___ Init GP ___
screening = DESilico.DictScreening(Dict(sequences .=> fitness), missing_fitness_value)

embedding_extractor = Dict(sequences_complete .=> sequence_embeddings)
_extract_embedding(domain_coords::AbstractVector{Float64}) = _extract_embedding(Vector{Int}(domain_coords))
function _extract_embedding(domain_coords::AbstractVector{Int})
    residues = _decode_domain(domain_coords)
    sequence = _construct_sequence(residues, wt_string, mutation_positions)
    embedding_extractor[sequence]
end

domain = Domain(;
    bounds=([1, 1, 1, 1], [20, 20, 20, 20]),
    discrete=[true, true, true, true],
)

model = EmbeddingGP(
    EmbeddingKernel(Matern32Kernel()), # Matern32Kernel(; metric=Euclidean())
    Product([truncated(Normal(0, sqrt(1280) / 3.0); lower=0.0)]), # Multivariate dist with one dimension 
    _extract_embedding,
)

data = BOSS.ExperimentDataPrior(Matrix{Float64}(hcat(_encode_domain_float(wt_variant))), hcat([screening(wt_sequence)]))
problem = BossProblem(;
    fitness=LinFitness([1]),
    f=x -> [screening(_construct_sequence(_decode_domain(x), wt_string, mutation_positions))],
    domain,
    model,
    noise_var_priors=[Dirac(0.0)],
    data,
)

model_fitter = BOSS.OptimizationMLE(;
    algorithm=NEWUOA(),
    multistart=20,
    parallel=false,
    rhoend=1e-4,
)

# ___ Run GP ___
bo!(problem;
    model_fitter,
    acq_maximizer=DEAcqMaximizer(variant_coords),
    acquisition=ExpectedImprovement(),
    term_cond=IterLimit(199),
    options=BossOptions(;
        info=true,
        debug=false,
    ),
)

save(
    joinpath(@__DIR__, "data", "gpde", "GB1", "01", "gp.jld2"),
    "problem", problem,
)
