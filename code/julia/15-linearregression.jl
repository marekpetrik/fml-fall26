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
println("Columns:", names(df))                                  # check the exact column names

us = dropmissing(filter(r -> r.country == "United States", df), :new_cases_smoothed)
y = us[1:400, :new_cases_smoothed]

plot(y, xlabel = "time", ylabel = "COVID cases")

### --- Construct features

n = length(y)
A = hcat(ones(n), 1:n)

f(x) = 1/2 * norm(A*x - y)^2  # 1/2 RSS


# the following block adds polynomial features
if false
    A = hcat(A, (1:n).^2)
end

ReLU(x) = max(x, 0)

# the following block adds splines
if false
    θ₁₀ = -100
    θ₁₁ = 1
    θ₂₀ = -200
    θ₂₁ = 1
    θ₃₀ = -300
    θ₃₁ = 1
    A = hcat(A, ReLU.(θ₁₁ * (1:n) .+ θ₁₀))
    A = hcat(A, ReLU.(θ₂₁ * (1:n) .+ θ₂₀))
    A = hcat(A, ReLU.(θ₃₁ * (1:n) .+ θ₃₀))
end


### --- Linear regression: normal equations

x̂ = (A'*A) \ (A'*y)              
println("Normal equations: x̂ = $x̂, f(x̂) = $(f(x̂))")

p = plot(y, xlabel="time", ylabel="COVID cases", label="data", color="black")
plot!(A*x̂, label="ŷ", color="red", width=2)

display(p)
#println("\npress enter to close plot")
#readline()

### --- Linear regression: gradient descent (manual)

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

∇f₁(x) = A'*(A*x - y)      # analytical gradient

# hand-selected stepsize
α = 0.00000001

# optimal step size (for the quadratic function)
#λs = eigen(2*A'*A).values
#α = 2/(maximum(λs) + minimum(λs))

K = 100000                 # number of iterations
x₀ = [100000.0,-1000.0]
#x₀ = [0.0,0.0]

xs₁ = gradient_descent(∇f₁, x₀, α, K)
println("GD (manual): x = $(xs₁[end]), f(x) = $(f(xs₁[end]))")

p = plot(y, xlabel="time", ylabel="COVID cases", label="data", color="black")
plot!(A*xs₁[1], label="ŷ₀", color="red", width=2, linestyle=:dash)
plot!(A*xs₁[div(length(xs₁),2)], label="ŷ₁₀", color="red", width=2, linestyle=:dash)
plot!(A*xs₁[end], label="ŷ", color="red", width=2)

display(p)
#println("\npress enter to close plot")
#readline()

### --- plot convergence

errors = [norm(x - x̂) for x ∈ xs₁]
plot(errors)

#maximum([errors[i+1] / errors[i] for i ∈ eachindex(errors) if i < length(errors)])

### --- Linear regression: gradient descent (autograd)

∇f₂(x) = gradient(f, AutoForwardDiff(), x)

xs₂ = gradient_descent(∇f₂, x₀, α, K)
println("GD (autograd): x = $(xs₂[end]), f(x) = $(f(xs₂[end]))")

p = plot(y, xlabel="time", ylabel="COVID cases", label="data", color="black")
plot!(A*xs₂[1], label="ŷ₀", color="red", width=2)
plot!(A*xs₂[11], label="ŷ₁₀", color="red", width=2)
plot!(A*xs₂[end], label="ŷ", color="red", width=2)

display(p)
println("\npress enter to close plot")
#readline()
