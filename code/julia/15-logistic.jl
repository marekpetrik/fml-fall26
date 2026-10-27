using Downloads, CSV, DataFrames, Plots
using LinearAlgebra
using Statistics                 # for the median

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
cases = us[1:400, :new_cases_smoothed]

### --- Binary labels: are the cases above the median?

threshold = median(cases)
y = Float64.(cases .> threshold)  # 1 = high, 0 = low

p = plot(cases, xlabel="time", ylabel="COVID cases", label="data", color="black")
hline!([threshold], label="median", color="red", linestyle=:dash, width=2)

display(p)
println("\npress enter to close plot")
readline()

### --- Logistic regression: Newton's method (NO NEED TO KNOW)

n = length(y)
t = (1:n) / n                    # rescale time to (0,1] to make gradient descent converge faster
A = hcat(ones(n), t)             # features: intercept and time

σ(z) = 1 / (1 + exp(-z))         # logistic function: P[yᵢ = 1] = σ(aᵢ'x)

f(x) = 1/n * sum(log1p.(exp.(A*x)) .- y .* (A*x))  # (mean) negative log-likelihood

∇f₁(x) = 1/n * A'*(σ.(A*x) - y)                          # analytical gradient
∇²f(x) = 1/n * A'*Diagonal(σ.(A*x) .* (1 .- σ.(A*x)))*A  # Hessian

# no closed form like the normal equations for linear regression;
# each Newton step solves weighted normal equations
function newton(∇f, ∇²f, x₀, K)
    x = x₀
    for k ∈ 1:K
        x = x - ∇²f(x) \ ∇f(x)
    end
    return x
end

x₀ = zeros(2)
x̂ = newton(∇f₁, ∇²f, x₀, 10)
println("Newton's method: x̂ = $x̂, f(x̂) = $(f(x̂))")

p = plot(y, xlabel="time", ylabel="cases > median", label="data", color="black")
plot!(σ.(A*x̂), label="logistic regression", color="red", width=2)

display(p)
println("\npress enter to close plot")
readline()

### --- Logistic regression: gradient descent (manual)

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

L = opnorm(A'*A) / (4n)          # Lipschitz constant of ∇f (since σ'(z) ≤ 1/4)
α = 1 / L                        # step size
K = 100                          # number of iterations

xs₁ = gradient_descent(∇f₁, x₀, α, K)
println("GD (manual): x = $(xs₁[end]), f(x) = $(f(xs₁[end]))")

### --- Logistic regression: gradient descent (autograd)

∇f₂(x) = gradient(f, AutoForwardDiff(), x)

xs₂ = gradient_descent(∇f₂, x₀, α, K)
println("GD (autograd): x = $(xs₂[end]), f(x) = $(f(xs₂[end]))")

# plot the evolution of the objective (gap to the optimum from Newton's method)
p = plot(0:K, f.(xs₁) .- f(x̂), xlabel="iteration", ylabel="f(x) - f(x̂)", yscale=:log10,
         label="GD (manual)", color="blue", width=2)
plot!(0:K, f.(xs₂) .- f(x̂), label="GD (autograd)", color="green", linestyle=:dash, width=2)

display(p)
println("\npress enter to close plot")
readline()
