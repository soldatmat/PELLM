using FileIO
using CSV
using DataFrames
using PyCall

using BOSS
using DESilico
using OptimizationPRIMA

py"""
import pickle
 
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""
load_pickle = py"load_pickle"

include("../de/utils.jl")
include("../de/boes/lib/embedding_gp.jl")
include("../de/boes/lib/de_acq_maximizer.jl")


# GB1
data_path = joinpath(@__DIR__, "..", "..", "data", "GB1")
wt_string = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"  # ['V', 'D', 'G', 'V']
mutation_positions = [39, 40, 41, 54]
missing_fitness_value = 0.0
neighborhoods_filename = "gb1_esm1b_euclidean.jld2"
#coord_filename = "GB1_pca.pkl"
coord_filename = "GB1_tsne.pkl"
coord_variants = "GB1_variants.pkl"

# PhoQ
#= data_path = joinpath(@__DIR__, "..", "..", "data", "PhoQ")
wt_string = "MKKLLRLFFPLSLRVRFLLATAAVVLVLSLAYGMVALIGYSVSFDKTTFRLLRGESNLFYTLAKWENNKLHVELPENIDKQSPTMTLIYDENGQLLWAQRDVPWLMKMIQPDWLKSNGFHEIEADVNDTSLLLSGDHSIQQQLQEVREDDDDAEMTHSVAVNVYPATSRMPKLTIVVVDTIPVELKSSYMVWSWFIYVLSANLLLVIPLLWVAAWWSLRPIEALAKEVRELEEHNRELLNPATTRELTSLVRNLNRLLKSERERYDKYRTTLTDLTHSLKTPLAVLQSTLRSLRSEKMSVSDAEPVMLEQISRISQQIGYYLHRASMRGGTLLSRELHPVAPLLDNLTSALNKVYQRKGVNISLDISPEISFVGEQNDFVEVMGNVLDNACKYCLEFVEISARQTDEHLYIVVEDDGPGIPLSKREVIFDRGQRVDTLRPGQGVGLAVAREITEQYEGKIVAGESMLGGARMEVIFGRQHSAPKDE"
mutation_positions = [284, 285, 288, 289]
missing_fitness_value = 0.0
neighborhoods_filename = "phoq_esm1b_euclidean.jld2"
#coord_filename = "PhoQ_pca.pkl"
coord_filename = "PhoQ_tsne.pkl"
coord_variants = "PhoQ_variants.pkl" =#

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

embedding_extractor = Dict(sequences_complete .=> sequence_embeddings)
_extract_embedding(domain_coords::AbstractVector{Float64}) = _extract_embedding(Vector{Int}(domain_coords))
function _extract_embedding(domain_coords::AbstractVector{Int})
    residues = _decode_domain(domain_coords)
    sequence = _construct_sequence(residues, wt_string, mutation_positions)
    embedding_extractor[sequence]
end

# ___ Load Results ___
gp_path = joinpath(@__DIR__, "..", "de", "data", "gpde")
gp_filename = "gp_384_no_repeats.jld2"
d = load(joinpath(gp_path, gp_filename))
problem = d["problem"]

model = EmbeddingGP(problem.model.kernel, problem.model.length_scale_prior, _extract_embedding)
posterior = BOSS.model_posterior(model, problem.data)

# ___ Load coordinates ___
x2c(x, cv::AbstractVector{Vector{Int64}}) = findfirst(a->a==x, cv)
function _index2coords(x_indexes::AbstractVector{Int64})
    x1 = map(x -> c[x,1], x_indexes)
    x2 = map(x -> c[x,2], x_indexes)
    return x1, x2
end
function get_x_coords(variant_codes::AbstractMatrix{T}, cv::AbstractVector{Vector{Int64}}) where {T <: Real}
    x_indexes = map(x -> x2c(x, cv), eachcol(variant_codes))
    return _index2coords(x_indexes)
end
function get_x_coords(variant_codes::AbstractVector{Vector{T}}, cv::AbstractVector{Vector{Int64}}) where {T <: Real}
    x_indexes = map(x -> x2c(x, cv), variant_codes)
    return _index2coords(x_indexes)
end
coord_path = joinpath(@__DIR__, "..", "dimred")
c = load_pickle(joinpath(coord_path, coord_filename))[:,1:2]
cv = load_pickle(joinpath(coord_path, coord_variants))
cv = map(c -> _encode_domain(collect(c)), cv)
x1, x2 = get_x_coords(problem.data.X, cv)
y = vec(problem.data.Y)

code_alphabet = collect(1:20)
combine(v...) = vec(collect(Iterators.product(v...)))
variant_codes = combine(code_alphabet, code_alphabet, code_alphabet, code_alphabet)
#variant_codes = sample(variant_codes, 10000, replace=false, ordered=false)
variant_codes = map(vc -> collect(vc), variant_codes)
x1s, x2s = get_x_coords(variant_codes, cv)
outputs = map(vc -> posterior(vc), variant_codes)
ys = map(o -> o[1][1], outputs)
vs = map(o -> o[2][1], outputs)

data_sample = map(i -> (x1s[i], x2s[i], ys[i], vs[i]), eachindex(x1s))
sort!(data_sample, by=x -> x[3])
x1s = map(d -> d[1], data_sample)
x2s = map(d -> d[2], data_sample)
ys = map(d -> d[3], data_sample)
vs = map(d -> d[4], data_sample)

# ___ Save stuff ___
pickle = pyimport("pickle")
file_path = joinpath(@__DIR__, "boes_gb1_tsne.pkl")
@pywith pybuiltin("open")(file_path, "wb") as f begin
    pickle.dump([
            x1s,
            x2s,
            ys,
        ], f)
end

# ___ Plot ___
using Plots
pythonplot()
plot(x1, x2, seriestype=:scatter, label="data")
plot(x1, y, seriestype=:scatter, label="data")

scatter(x1, x2; zcolor=y)
scatter(x1s, x2s; zcolor=ys)

x1_grid = range(-0.8, 1.0, length=90)
x2_grid = range(-1.0, 1.25, length=90)
y_grid = @. posterior(x1_grid', x2_grid)

contour(x1, x2, y)
contourf(x1, x2, y, levels=20, color=:turbo)

py"""
import matplotlib.pyplot as plt

def plot_scatter(x1, x2, y):
    plt.scatter(
        x1,
        x2,
        marker=".",
        #s=150,
        #linewidths=4,
        #c=fitness,
        #c=fitness[0:160000],
        c=y,
        cmap=plt.cm.inferno_r,
        vmin=0.,
        #vmax=133.59427,
        vmax=60.,
    )
    plt.colorbar(label="Fitness")
    plt.xlabel("PC1 22.9% variance")
    plt.ylabel("PC2 15.4% variance")
    plt.show()
"""
plot_scatter = py"plot_scatter"

GB1_MAX_FITNESS = 8.76196565571
PHOQ_MAX_FITNESS = 133.59427
plot_scatter(x1s, x2s, ys .* PHOQ_MAX_FITNESS)
