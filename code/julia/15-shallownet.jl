using Downloads, CSV, DataFrames, Plots
using LinearAlgebra

using DifferentiationInterface # see: https://github.com/JuliaDiff/DifferentiationInterface.jl
using ForwardDiff: ForwardDiff # AD backend

### --- Download dataset

url  = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
path = "owid_covid.csv"                            # saved in the current directory

if isfile(path)
    println("Using cached data in $path")
else
    println("Fetching data from $url ...")
    Downloads.download(url, path)
end

df = CSV.read(path, DataFrame)
println("Data dimensions: ", size(df))

us = dropmissing(filter(r -> r.country == "United States", df), :new_cases_smoothed)
y = us[1:400, :new_cases_smoothed] / 100_000      # in 100k cases

plot(y, xlabel = "time", ylabel = "COVID cases (100k)")

### --- Shallow network: one hidden layer with ReLU activations

# gradient descent needs all parameters on a similar scale: rescale time (and cases above)
n = length(y)
t = (1:n) / n                    # time in (0,1]

ReLU(z) = max(z, 0)

# hidden units: hⱼ(t) = ReLU(θⱼ₁ t + θⱼ₀),  output: ŷ(t) = w₀ + Σⱼ wⱼ hⱼ(t)
# with θⱼ₁ = 1 and θⱼ₀ = -knotⱼ, the hidden units are the spline features
knots = [0, 100, 200, 300] / n   # in days; the knot at 0 is the linear term: ReLU(t) = t
m = length(knots)                # number of hidden units

# all parameters in one vector: x = [w₀; w; θ₀; θ₁]
unpack(x) = (x[1], x[2:m+1], x[m+2:2m+1], x[2m+2:3m+1])

function network(x)
    w₀, w, θ₀, θ₁ = unpack(x)
    H = ReLU.(t * θ₁' .+ θ₀')     # hidden layer: Hᵢⱼ = hⱼ(tᵢ)
    return w₀ .+ H * w            # output layer
end

f(x) = 1/(2n) * norm(network(x) - y)^2  # (half) mean squared error

### --- Spline regression: normal equations (fixed knots)

θ₀ = -knots
θ₁ = ones(m)
A = hcat(ones(n), ReLU.(t * θ₁' .+ θ₀'))   # features: intercept and splines

ŵ = (A'*A) \ (A'*y)              # solve the normal equations A'A w = A'y
x̂ = [ŵ; θ₀; θ₁]
println("Spline (normal equations): f(x̂) = $(f(x̂))")

p = plot(y, xlabel="time", ylabel="COVID cases (100k)", label="data", color="black")
plot!(network(x̂), label="ŷ (spline)", color="red", width=2)

display(p)
println("\npress enter to close plot")
#readline()

### --- Shallow network: gradient descent (autograd)

# returns the iterates x₀, x₁, …, x_K
function gradient_descent(∇f, x₀, α, K)
    x = x₀
    xs = [x]
    for k ∈ 1:K
        x = x - α * ∇f(x)
        push!(xs, x)
    end
    return xs
end

∇f(x) = gradient(f, AutoForwardDiff(), x)

α = 0.05                         # step size; α = 0.5 is unstable at first, but ends in a better local minimum
K = 5_000                        # number of iterations
x₀ = x̂                           # start from the spline; GD also moves the knots

xs = gradient_descent(∇f, x₀, α, K)
println("GD (autograd): f(x) = $(f(xs[end]))")

_, _, θ₀, θ₁ = unpack(xs[end])
println("Learned knots (days): ", round.(Int, -θ₀ ./ θ₁ * n))

# iterate xₖ is xs[k+1]
p = plot(y, xlabel="time", ylabel="COVID cases (100k)", label="data", color="black")
plot!(network(xs[1]), label="ŷ₀ (spline)", color="blue", linestyle=:dash, width=2)
plot!(network(xs[101]), label="ŷ₁₀₀", color="green", linestyle=:dash, width=2)
plot!(network(xs[end]), label="ŷ", color="red", width=2)

display(p)
println("\npress enter to close plot")
#readline()

# plot the evolution of the objective
p = plot(0:K, f.(xs), xlabel="iteration", ylabel="f(x)", label="GD (autograd)",
         color="red", width=2)
hline!([f(x̂)], label="spline", color="blue", linestyle=:dash, width=2)

display(p)
println("\npress enter to close plot")
#readline()
