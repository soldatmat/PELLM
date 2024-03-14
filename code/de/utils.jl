torch = pyimport("torch")

function _tensor_probabilities_to_touples(probabilities::PyObject, alphabet::Vector{Char})
    probabilities = probabilities.cpu().numpy() # PyObject tensor -> Matrix{Float32}
    probabilities = [probabilities[seq, :] for seq in 1:size(probabilities)[1]] # Matrix{Float32} -> Vector{Vector{Float32}}
    [[(alphabet[symbol], pdist[symbol]) for symbol in eachindex(alphabet)] for pdist in probabilities] # Vector{Vector{Float32}} -> Vector{Dict{Char, Float32}}
end

default_torch_device() = torch.device(torch.cuda.is_available() ? "cuda" : "cpu")

tensor_to_matrix(tensor::PyObject) = tensor.cpu().numpy()

ensure_tensor(data::Any) = pybuiltin("type")(data) == torch.Tensor ? data : torch.tensor(data)
