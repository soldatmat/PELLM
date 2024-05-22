using DESilico

struct TrainingScreening <: DESilico.Screening
    screening::DESilico.Screening
    train::Function
end

function (s::TrainingScreening)(sequences::AbstractVector{Vector{Char}})
    fitness = s.screening(sequences)
    s.train(map(i -> Variant(sequences[i], fitness[i]), eachindex(sequences)))
    return fitness
end
