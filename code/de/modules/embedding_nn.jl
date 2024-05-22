using PyCall

torch = pyimport("torch")

pushfirst!(pyimport("sys")."path", joinpath(@__DIR__, "python"))
embedding_nn = pyimport("embedding_nn")

"""
TODO

    EmbeddingNN(embedding_extractor::AbstractEmbeddingExtractor, model::PyObject, device::PyObject)
    EmbeddingNN(embedding_extractor::T, model::PyObject; device::PyObject=_default_device())

# Arguments
- `embedding_extractor::AbstractEmbeddingExtractor`: Structure used to extract embeddings from sequences. See `types.jl`.
- `model::PyObject`: `torch.nn.Module` object. Model which takes sequence embedding as input and outputs predicted fintess.
- `device::PyObject`: `torch.device` object. Defines the device where the model will be loaded.
"""
struct EmbeddingNN{T<:AbstractEmbeddingExtractor} <: FitnessPredictor
    embedding_extractor::T
    model::PyObject
    device::PyObject

    EmbeddingNN(embedding_extractor::T, model::PyObject, device::PyObject) where {T<:AbstractEmbeddingExtractor} = new{T}(embedding_extractor, model.to(device), device)
end

EmbeddingNN(embedding_extractor, model; device=default_torch_device()) = EmbeddingNN(embedding_extractor, model, device)

function (fp::EmbeddingNN)(sequences::AbstractVector{Vector{Char}})
    println("EmbeddingNN predict: Extracting sequence embeddings with $(typeof(fp.embedding_extractor)) ...")
    embeddings = fp.embedding_extractor(sequences)
    embeddings = _ensure_model_dtype(fp.model, ensure_tensor(embeddings))
    println("EmbeddingNN predict: Predicting fitness with $(t=pybuiltin("type")(fitness_predictor.model)) ...")
    fp.model.eval()
    torch.set_grad_enabled(false)
    predictions = fp.model(embeddings.to(fp.device))
    torch.set_grad_enabled(true)
    GC.gc() # GPU allocs by Python sometimes do not get freed automatically
    println("EmbeddingNN predict: Finished.")
    return predictions
end

function train!(fp::EmbeddingNN, variants::AbstractVector{Variant})
    println("EmbeddingNN train: Extracting sequence embeddings with $(typeof(fp.embedding_extractor)) ...")
    embeddings = fp.embedding_extractor([v.sequence for v in variants])
    println("EmbeddingNN train: Training model ...")
    embeddings = _ensure_model_dtype(fp.model, ensure_tensor(embeddings))
    fitness_values = [v.fitness for v in variants]
    fitness_values = _ensure_model_dtype(fp.model, ensure_tensor(fitness_values))
    embedding_nn.train(fp.model, embeddings, fitness_values, fp.device)
    println("EmbeddingNN train: Finished.")
end

function _ensure_model_dtype(model::PyObject, data::PyObject)
    model_dtype = [p for p in model.parameters()][1].dtype
    data.to(model_dtype)
end
