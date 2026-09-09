using CSV         # https://csv.juliadata.org/stable/reading.html
using DataFrames  # https://dataframes.juliadata.org/stable/

using LinearAlgebra # For the dot product

### Load the matrix
# See: https://csv.juliadata.org/stable/reading.html#CSV.read
#      https://csv.juliadata.org/stable/examples.html#matrix_example
F = CSV.File("advertising.csv")
A = F |> DataFrame |> Matrix


### Compute the dot product between vectors
v = A[:,1] 
w = A[:,2]

println(v ⋅ w)


### Compute the correlation coefficient
using StatsBase

println(cor(v,w))

# center features (there is a more elengant way too!)
n = length(v)
v = v - (ones(n)/n) ⋅ v * ones(n)
w = w - (ones(n)/n) ⋅ w * ones(n)

# check that they are centered
sum(v)
sum(w)

# compute the correlation coefficient

v ⋅ w / norm(v) / norm(w)


### Normalize the features to sum to one

A₂ = A * diagm(1 ./ (A' * ones(n)))

A₂' * ones(n)
