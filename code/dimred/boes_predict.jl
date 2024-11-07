using FileIO
using CSV
using DataFrames
using PyCall

using BOSS
using DESilico
using OptimizationPRIMA

include("../de/utils.jl")
include("../de/boes/lib/embedding_gp.jl")
include("../de/boes/lib/de_acq_maximizer.jl")

# ___ Choose run ___
dataset_name = "GB1" # "GB1", "PhoQ"
run_folder = "BOSS_v0.2.0_24_budget"
gp_filename = "boes_wt.jld2"

# ___ Load Results ___
gp_path = joinpath(@__DIR__, "..", "de", "data", "boes", dataset_name, run_folder)
d = load(joinpath(gp_path, gp_filename))
problem = d["problem"]

model = EmbeddingGP(
    problem.model.kernel,
    problem.model.length_scale_prior,
    _extract_embedding,
    problem.model.noise_std_priors
)
posterior = BOSS.model_posterior(model, problem.data)

# ___ Evaluate variants ___
code_alphabet = collect(1:20)
combine(v...) = vec(collect(Iterators.product(v...)))
variant_codes = combine(code_alphabet, code_alphabet, code_alphabet, code_alphabet)
#variant_codes = sample(variant_codes, 10000, replace=false, ordered=false)
variant_codes = map(vc -> collect(vc), variant_codes)
outputs = map(vc -> posterior(vc), variant_codes)
ys = map(o -> o[1][1], outputs)
vs = map(o -> o[2][1], outputs)

# ___ BOES domain to residue conversion ___
alphabet = collect(DESilico.alphabet.protein)
domain_encoder = Dict(alphabet .=> eachindex(alphabet))
_encode_domain_float(variant::AbstractVector{Char}) = map(symbol -> Float64.(domain_encoder[symbol]), variant)
_encode_domain(variant::AbstractVector{Char}) = map(symbol -> domain_encoder[symbol], variant)
domain_decoder = Dict(eachindex(alphabet) .=> alphabet)
_decode_domain(coords::AbstractVector{Int}) = map(coord -> domain_decoder[coord], coords)

variant_residues = String.(_decode_domain.(variant_codes))

# ___ Save predictions ___
pickle = pyimport("pickle")
prediction_filename = "variant_prediction.pkl"
file_path = joinpath(gp_path, prediction_filename)
@pywith pybuiltin("open")(file_path, "wb") as f begin
    pickle.dump([
            ys,
            vs,
            variant_residues,
        ], f)
end
