using PyCall

torch = pyimport("torch")
esm = pyimport("esm")

pushfirst!(pyimport("sys")."path", joinpath(@__DIR__, "..", "python"))
mlm = pyimport("mlm")

"""
`LLM` interface for ESM-1b model.
git: https://github.com/facebookresearch/esm
paper: https://www.biorxiv.org/content/10.1101/622803v4

    ESM1b(device::PyObject)
    ESM1b(; device::PyObject=_default_device())

# Arguments
- `device::PyObject`: `torch.device` object. Defines the device where the model will be loaded.
"""
struct ESM1b <: LLM
    model::PyObject
    embedding_size::Int
    device::PyObject
    alphabet::PyObject
    batch_converter::PyObject

    function ESM1b(device)
        model, alphabet = _init_model(device)
        batch_converter = alphabet.get_batch_converter()
        new(model, 1280, device, alphabet, batch_converter)
    end

    function _init_model(device::PyObject)
        println("Initializing ESM1b model")
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        model = model.to(device)
        return model, alphabet
    end
end

ESM1b(; device=default_torch_device()) = ESM1b(device)

tokenize(llm::ESM1b, sequence::String) = llm.batch_converter([(nothing, sequence)])[3]
tokenize(llm::ESM1b, sequence::AbstractVector{Char}) = tokenize(llm, String(sequence))
tokenize(llm::ESM1b, sequences::AbstractVector{String}) = llm.batch_converter(map(s -> (nothing, s), sequences))[3]
tokenize(llm::ESM1b, sequences::AbstractVector{Vector{Char}}) = tokenize(llm, map(s -> String(s), sequences))

# TODO ? add 1000 extra randomly selected sequences from the UniRef50 database (relative to 8000 best-predicted GB1 sequences)
function train!(llm::ESM1b, sequences::AbstractVector{Vector{Char}}; mask_positions::AbstractVector{Int}=nothing)
    println("ESM1b train: Training...")
    if isnothing(mask_positions)
        println("ESM-1b training without mask positions NOT IMPLEMENTED.")
    else
        # mask_positions are shifted by 0 = +1 (ESM-1b tokenize) -1 (Julia -> Python indexing)
        mlm.train(llm.model, tokenize(llm, sequences), mask_positions, llm.alphabet.mask_idx, length(llm.alphabet.all_toks), llm.device)
    end
    GC.gc() # GPU allocs by Python sometimes do not get freed automatically
    println("ESM1b train: Finished.")
end

"""
Returns PyObject tensor with size (length of `batch`, length of longest sequence in `batch`, 1280).
"""
function extract_embedding(llm::ESM1b, batch::PyObject)
    llm.model.eval()
    torch.set_grad_enabled(false)
    output = llm.model(batch.to(llm.device), repr_layers=[33])
    torch.set_grad_enabled(true)
    output["representations"][33]
end

"""
Returns PyObject tensor with size (length of `batch`, 1280).
"""
function extract_sequence_embedding(llm::ESM1b, batch::PyObject)
    embedding = extract_embedding(llm, batch)
    embedding = embedding.narrow(1, 1, embedding.size()[2] - 2).mean(1) # arguments use Python indexing
end

"""
Returns PyObject tensor with size (length of `batch`, 1280).
"""
function extract_contextualized_embedding(llm::ESM1b, batch::PyObject, token_index::Int)
    embedding = extract_embedding(llm, batch)
    # token_index is shifted by 0 = +1 (ESM-1b tokenize) -1 (Julia -> Python indexing)
    py"""_select_token_embedding"""(embedding, token_index)
end
py"""
def _select_token_embedding(embedding, token_index):
    return embedding[:, token_index, :]
"""

function mask_sequence!(llm::ESM1b, sequence::PyObject, mask_index::Int)
    sequence[mask_index+1] = llm.alphabet.mask_idx # token_index is shifted by 1 = +1 (ESM-1b tokenize)
end

# TODO test
# TODO check correct indexing with `mask_positions`
# function get_pseudolikelihoods(llm::ESM1b, masked_sequence::Vector{Float32}, mask_positions::Vector{Int})
#     llm.model.eval()
#     torch.set_grad_enabled(false)

#     masked_sequence = torch.tensor([masked_sequence]).to(llm.device)
#     output = llm.model(masked_sequence)
#     mask_logits = py"_select_mask_logits"(output, mask_positions .- 1) # -1 for Python indexing
#     pseudolikelihoods = torch.nn.functional.sigmoid(mask_logits)
#     pseudolikelihoods = _tensor_probabilities_to_dicts(pseudolikelihoods, llm.alphabet)

#     torch.set_grad_enabled(true)
#     return pseudolikelihoods
# end
# py"""
# def _select_mask_logits(output, mask_positions):
#     return output["logits"][0, mask_positions, :]
# """
