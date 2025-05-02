using DESilico
using Distributions
using AbstractGPs
using LinearAlgebra

struct EmbeddingKernel <: Kernel
    kernel::Kernel
end

function (kernel::EmbeddingKernel)(emb1, emb2)
    kernel.kernel(emb1, emb2)
end

struct AdditiveKernel <: Kernel
    kernel::Kernel
end

function (kernel::AdditiveKernel)(emb1, emb2)
    sum((sqrt(1280) / 3)^2 .* [kernel.kernel(emb1[d], emb2[d]) for d in length(emb1)])
end

struct ConicKernel <: Kernel
    r_kernel::Kernel
    kernel::Kernel
end

function (kernel::ConicKernel)(emb1, emb2)
    norm1 = norm(emb1)
    norm2 = norm(emb2)
    e1 = emb1 ./ norm1
    e2 = emb2 ./ norm2
    a = 5
    b = 5

    one = 1 - real(complex(1 - real(complex(norm1)^a))^b)
    two = 1 - real(complex(1 - real(complex(norm2)^a))^b)

    return kernel.r_kernel(one, two) * kernel.kernel(e1, e2)
end

"""
Defined for completeness but not called in BOES procedure.

Because `finite_EmbeddingGP` returns `BOSS.finite_gp(...)` so
`KernelFunctions.with_lengthscale(dk::DiscreteKernel, lengthscale::Real)`,
which is defined in BOSS, is called instead.
"""
function with_lengthscale(kernel::EmbeddingKernel, lengthscale::Real)
    EmbeddingKernel(KernelFunctions.with_lengthscale(kernel.kernel, lengthscale))
end
function with_lengthscale(kernel::EmbeddingKernel, lengthscales::AbstractVector{<:Real})
    @assert length(lengthscales) == 1
    with_lengthscale(kernel, lengthscales[1])
end

# kernel = with_lengthscale(SqExponentialKernel(), 2.5);
# kernel(x, y) ≈ (SqExponentialKernel() ∘ ScaleTransform(0.4))(x, y)    = SqExponentialKernel(x .* 0.4, y .* 0.4)
#
# kernel = with_lengthscale(SqExponentialKernel(), [0.5, 2.5]);
# kernel(x, y) ≈ (SqExponentialKernel() ∘ ARDTransform([2, 0.4]))(x, y) = SqExponentialKernel(x .* [2, 0.4], y .* [2, 0.4])

struct EmbeddingGP <: BOSS.SurrogateModel
    kernel::Union{EmbeddingKernel,AdditiveKernel,ConicKernel}
    length_scale_prior::MultivariateDistribution
    extract_embedding::Function
    noise_std_priors::AbstractVector{<:UnivariateDistribution}
end

function finite_EmbeddingGP(X::AbstractMatrix{<:Real}, kernel::Kernel, length_scales::AbstractVector{<:Real}, noise_std::Real)
    mean = nothing
    amplitude = 1.0
    BOSS.finite_gp(X, mean, kernel, length_scales, amplitude, noise_std)
end

BOSS.make_discrete(model::EmbeddingGP, discrete::AbstractVector{<:Bool}) = model

_extract_embeddings(X::AbstractMatrix{<:Real}) = reduce(hcat, model.extract_embedding.(eachcol(X)))

function BOSS.model_posterior(model::EmbeddingGP, data::BOSS.ExperimentDataMAP)
    θ, λ, α, noise_std = data.params
    embedding_posterior = AbstractGPs.posterior(
        finite_EmbeddingGP(
            _extract_embeddings(data.X),
            model.kernel,
            λ[:, 1],
            noise_std[1]
        ),
        data.Y[1, :],
    )

    function posterior(x)
        function fix_negative_var(var::Real)
            (var < 0.) & (var > -1e-10) ? 0. : var
        end

        x = model.extract_embedding(x)
        μ, σ2 = mean_and_var(embedding_posterior(hcat(x)))
        σ2 = fix_negative_var.(σ2)
        return μ, sqrt.(σ2)
    end
end

function BOSS.model_loglike(model::EmbeddingGP, data::ExperimentData)
    X = _extract_embeddings(data.X)
    function loglike(model_params::BOSS.ModelParams)
        θ, λ, α, noise_std = model_params
        ll_length_scales = logpdf(model.length_scale_prior, λ[:, 1])
        ll_noise_std = logpdf(model.noise_std_priors[1], noise_std[1])

        gp = finite_EmbeddingGP(X, model.kernel, λ[:, 1], noise_std[1])
        ll_data = logpdf(gp, data.Y[1, :])

        ll_noise_std + ll_length_scales + ll_data
    end
end

function BOSS.sample_params(model::EmbeddingGP)
    θ_sample = Real[]
    λ_sample = hcat(rand(model.length_scale_prior))
    α_sample = Real[]
    noise_std = rand.(model.noise_std_priors)
    return (θ_sample, λ_sample, α_sample, noise_std)
end

function BOSS.param_priors(model::EmbeddingGP)
    θ_priors = UnivariateDistribution[]
    λ_priors = [model.length_scale_prior]
    α_priors = UnivariateDistribution[]
    return (θ_priors, λ_priors, α_priors, model.noise_std_priors)
end
