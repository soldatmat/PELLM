scatter(fitness_complete, results, xlabel="starting fitness", ylabel="resulting fitness")

histogram2d(
    fitness_complete, results,
    xlabel="starting fitness",
    ylabel="resulting fitness",
    bins=(20, 20),
    clim=(0, 500),)
