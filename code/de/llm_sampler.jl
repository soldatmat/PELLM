"""
Uses provided sequences to finetune `llm` and chooses alphabets based on pseudolikelihoods
of tokens at `mask_positions`. Works only with LLMs for masked language modelling.

    LLMSampler(llm::LLM, masked_sequence::PyObject, alphabet::Set{Char}, k::Int)

# Arguments
- `llm::LLM`: Large language model used to obtain the pseudolikelihoods of tokens.
- `masked_sequence::PyObject`: Masked sequence tokenized for `llm` as PyObject tensor.
- `mask_positions::Vector{Int}`: Positions of mask tokens in `masked_sequence`.
- `alphabet::Set{Char}`: Defines symbols which can be used in place of masks.
- `k::Int`: Number of symbols which will be chosen at each mask position.

    LLMSampler(llm::LLM, tokenized_sequence::Vector{Int}, alphabet::Set{Char}, k::Int)

# Arguments
- `llm::LLM`: Large language model used to obtain the pseudolikelihoods of tokens.
- `tokenized_sequence::Vector{Int}`: Masked sequence tokenized for `llm`.
- `mask_positions::Vector{Int}`: Positions of mask tokens in `masked_sequence`.
- `alphabet::Set{Char}`: Defines symbols which can be used in place of masks.
- `k::Int`: Number of symbols which will be chosen at each mask position.

    LLMSampler(llm::LLM; masked_sequence::Vector{Char}, mask_token::Int, alphabet::Set{Char}, k::Int)

Extracts `mask_positions` from the `masked_sequence` String and tokenizes `masked_sequence`
with `llm`. Only works with `llm`s which use "<mask>" to signify the mask token.

# Arguments
- `llm::LLM`: Large language model used to obtain the pseudolikelihoods of tokens.

# Keywords
- `masked_sequence::Vector{Char}`: Masked sequence which will be tokenized with `llm`.
- `mask_token::Int`: Token used to represent mask by `llm`.
- `alphabet::Set{Char}`: Defines symbols which can be used in place of masks.
- `k::Int`: Number of symbols which will be chosen at each mask position.
"""
struct LLMSampler{T<:LLM} <: DESilico.AbstractAlphabetExtractor
    llm::T
    sampling_sequence::Vector{Char}
    alphabet::Vector{Char}
    k::Int

    LLMSampler(llm::T, sampling_sequence::Vector{Char}, alphabet::Set{Char}, k::Int) where {T<:LLM} = new{T}(llm, sampling_sequence, [symbol for symbol in alphabet], k)
end

LLMSampler(llm; sampling_sequence, alphabet, k) = LLMSampler(llm, sampling_sequence, alphabet, k)

# TODO implement pseudolikelihoods as an alternative
# pseudolikelihoods = get_pseudolikelihoods(ae.llm, ae.masked_sequence, ae.mask_positions)
function (ae::LLMSampler)(sequences::AbstractVector{Vector{Char}}, positions::AbstractVector{Int})
    train!(ae.llm, sequences; mask_positions=positions) # TODO enable llm training
    masks = _get_contextualized_mask_embeddings(ae, positions)
    symbols = _get_symbol_embeddings(ae)
    probabilities = _get_symbol_probability_distributions(ae, masks, symbols)
    _construct_alphabets!(ae, probabilities)
end

function _get_symbol_probability_distributions(ae::LLMSampler, masks::PyObject, symbols::PyObject)
    masks = torch.nn.functional.normalize(masks, p=2.0, dim=1)
    symbols = torch.nn.functional.normalize(symbols, p=2.0, dim=1)
    symbols = torch.transpose(symbols, 0, 1)
    p = torch.matmul(masks, symbols)
    p = torch.exp(p)
    p = p / p.sum(-1).unsqueeze(-1)
    _tensor_probabilities_to_touples(p, ae.alphabet)
end
function _get_symbol_embeddings(ae::LLMSampler)
    tokenized_symbols = tokenize(ae.llm, map(symbol -> [symbol], ae.alphabet))
    extract_contextualized_embedding(ae.llm, tokenized_symbols, 1)
end
function _get_contextualized_mask_embeddings(ae::LLMSampler, mask_positions::AbstractVector{Int})
    masked_sequence = tokenize(ae.llm, ae.sampling_sequence).squeeze()
    mask_sequence!(ae.llm, masked_sequence, mask_positions)
    mask_embeddings = Vector{PyObject}(undef, length(mask_positions))
    for m in eachindex(mask_positions)
        mask_embeddings[m] = extract_contextualized_embedding(ae.llm, masked_sequence.unsqueeze(0), mask_positions[m])
    end
    torch.stack(mask_embeddings).squeeze(1)
end

# TODO change Vector{Vector{...}} to Array{...}
function _construct_alphabets!(ae::LLMSampler, probabilities::Vector{Vector{Tuple{Char,Float32}}})
    alphabets = Vector{Set{Char}}(undef, length(probabilities))
    for pos in eachindex(probabilities)
        sort!(probabilities[pos], by=x -> x[2], rev=true)
        alphabets[pos] = Set(map(x -> x[1], probabilities[pos][1:ae.k]))
    end
    _print_sampled_alphabets(alphabets)
    return alphabets
end
function _print_sampled_alphabets(alphabets::Vector{Set{Char}})
    println("Alphabets sampled with LLM are: $(alphabets)")
end

# ___ OTD-based sampling ___

# unbalancedot = pyimport("unbalanced-ot-functionals.unbalancedot")

# function _construct_alphabets!(ae::LLMSampler, masks::PyObject, symbols::PyObject, probabilities::Vector{Vector{Tuple{Char,Float32}}})
#     otd = _calculate_otd(ae.k, masks, symbols, probabilities)

#     alphabets = Vector{Set{Char}}(undef, length(probabilities))
#     for pos in 1:length(probabilities)

#     end
#     _print_sampled_alphabets(alphabets)
#     return alphabets
# end
# # TODO implement OTD from afp-de
# # TODO add argument types
# function _calculate_otd(k::Int, masks, symbols, probabilities)
#     cost_matrix = _calculate_cost_matrix(symbols)
#     cost(x, y) = cost_matrix[x, y]
#     entropy = unbalancedot.entropy.KullbackLeibler(1e-2, 0.3) # TODO optimize parameters
#     for pos in eachindex(probabilities)
#         b = map(p -> p[2], probabilities[pos])
#         y = collect(eachindex(probabilities[pos]))
#         x = y
#         for symbol_selection in Iterators.product(y, k)
#             a = zeros(length(probabilities[pos]))
#             a
#         end
#     end
# end
# function _calculate_cost_matrix(symbols::PyObject)
#     n_symbols = symbols.size(0)
#     sym1 = symbols.repeat((n_symbols, 1))
#     sym2 = symbols.repeat_interleave(n_symbols, dim=0)
#     distance = torch.sqrt(torch.sum(torch.pow(torch.subtract(sym1, sym2), 2), dim=1))
#     torch.reshape(distance, (n_symbols, n_symbols))
# end
