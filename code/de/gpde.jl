using DESilico
using CSV
using DataFrames
using FileIO
using BOSS

function main()
    # TODO load data for screening init
    screening = DESilico.DictScreening(Dict(sequences .=> fitness), default_fitness_value)

    domain = Domain(;
        bounds=([1, 1, 1, 1], [20, 20, 20, 20]),
        discrete=[true, true, true, true],
    )

    model = EmbeddingGP(
        kernel=EmbeddingKernel(Matern32Kernel()),
        length_scale_prior=#TODO, # Multivariate dist with one dimension 
        extract_embedding=#TODO,
    )

    problem = BossProblem(;
        fitness=LinFitness([1]),
        f=x -> [screening(x)],
        domain,
        model,
        noise_var_priors=[Dirac(0.)],
        data=BOSS.ExperimentDataPrior(hcat(symbol2integer(wt_variant)), hcat([wt_variant_fitness])), # TODO
    )

    bo!(problem;
        acq_maximizer=AcqMaximizer,
        acquisition=ExpectedImprovement(),
        term_cond=IterLimit(1),
        options=BossOptions(;
            info=true,
            debug=false,
        ),
    )
end
