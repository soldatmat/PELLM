using DESilico
using Distributions
using AbstractGPs



struct EmbeddingKernel <: Kernel
    kernel::Kernel
end

function (kernel::EmbeddingKernel)(emb1, emb2)
    kernel.kernel(emb1, emb2; metric=Euclidean())
end

function with_lengthscale(kernel::EmbeddingKernel, lengthscale::Real)
    EmbeddingKernel(with_lengthscale(kernel.kernel, lengthscale))
end
function with_lengthscale(kernel::EmbeddingKernel, lengthscales::AbstractVector{<:Real})
    @assert length(lengthscales) == 1
    with_lengthscale(kernel, lengthscales[1])
end



struct EmbeddingGP
    kernel::EmbeddingKernel
    length_scale_prior::MultivariateDistribution
    extract_embedding::Function
end

make_discrete(model::EmbeddingGP, discrete::AbstractVector{<:Bool}) = model

function model_posterior(model::EmbeddingGP, data::ExperimentDataMLE)
    embedding_posterior = BOSS.model_posterior(
        nothing,
        model.kernel,
        reduce(hcat, model.extract_embedding.(eachcol(data.X))),
        data.Y[1,:],
        data.length_scales[:,1],
        data.noise_vars,
    )

    function posterior(x)
        x = model.extract_embedding(x)
        μ, σ2 = embedding_posterior(x)
        ([μ], [σ2])
    end
end

function model_loglike(model::EmbeddingGP, noise_var_priors::AbstractVector{<:UnivariateDistribution}, data::ExperimentData)
    X = reduce(hcat, model.extract_embedding.(eachcol(data.X)))
    function loglike(θ, length_scales, noise_vars)
        ll_noise_variance = logpdf(data.noise_var_priors[1], noise_vars[1])
        ll_length_scales = logpdf(data.length_scale_priors[1], length_scales[:,1])
        gp = BOSS.finite_gp(nothing, model.kernel, X, length_scales[:,1], noise_vars[1])
        ll_data = logpdf(gp, data.Y[1,:])
        ll_noise_variance + ll_length_scales + ll_data
    end
end

function sample_params(model::EmbeddingGP, noise_var_priors::AbstractVector{<:UnivariateDistribution})
    (θ_sample::AbstractVector{<:Real}, λ_sample::AbstractMatrix{<:Real}, noise_vars_sample::AbstractVector{<:Real})
    θ_sample = Real[]
    λ_sample = hcat(rand(model.length_scale_prior))
    noise_vars_sample = rand.(noise_var_priors)
    return (θ_sample, λ_sample, noise_vars_sample)
end

function param_priors(model::EmbeddingGP)
    θ_priors = UnivariateDistribution[]
    λ_priors = [model.length_scale_prior]
    return (θ_priors, λ_priors)
end
