# An example of using automatic differentiation 
# How AutoDiff works: https://arxiv.org/abs/2501.14787
# For some limitations, see:
# https://juliadiff.org/ForwardDiff.jl/stable/user/limitations/

using DifferentiationInterface # see: https://github.com/JuliaDiff/DifferentiationInterface.jl
using Plots
using LinearAlgebra

using ForwardDiff: ForwardDiff
using Enzyme: Enzyme
using Zygote: Zygote  # AD backends you want to use

### Experiments with projections 

A = [3.0 1; 1 2]
b = [1.0,2]

x̂ = A \ b
p = A * x̂

### Projections as minimizing a norm (differentiability issues)

f(x) = 1/2*norm(A * x - b)^2
#f(x) = 1/2*x'*A'*A*x - x'*A'*b + 1/2*b'*b

## Plots the function 
x₁ = range(-5, 5, 100)
x₂ = range(-5, 5, 100)
z = f.([[a₁, a₂] for a₁ ∈ x₁, a₂ ∈ x₂])

contour(x₁, x₂, z', xlabel="x₁", ylabel="x₂", aspect_ratio=:equal, size=(500,500))
scatter!([x̂[1]], [x̂[2]], label="x̂")  # minimum

# plot the gradient
∇f1(x) = A'*A*x - A'*b
∇f2(x) = value_and_gradient(f, AutoForwardDiff(), x)[2]
∇f3(x) = value_and_gradient(f, AutoZygote(), x)[2]
∇f4(x) = value_and_gradient(f, AutoEnzyme(), x)[2]
x = [0.0,0]
∇fx = ∇f1(x)

scatter!([x[1]], [x[2]], label="x", color=:red)  # eval point
plot!([x[1],x[1]+0.1*∇fx[1]],[x[2],x[2]+0.1*∇fx[2]],arrow=true,color=:red, linewidth=2,
       label="0.1⋅∇f(x)")

println("Analytical: $(∇f1(x))")
println("ForwardDiff: $(∇f2(x))")
println("Zygote: $(∇f3(x))")
println("Enzyme: $(∇f4(x))")

### Experiments with another function

f(x) = log(1 + x'*x)

## Plots the function 
x₁ = range(-5, 5, 100)
x₂ = range(-5, 5, 100)
z = f.([[a₁, a₂] for a₁ ∈ x₁, a₂ ∈ x₂])

contour(x₁, x₂, z', xlabel="x₁", ylabel="x₂", aspect_ratio=:equal, size=(500,500))

# plot the gradient
∇f1(x) = 2/(1 + x'*x) * x
∇f2(x) = value_and_gradient(f, AutoForwardDiff(), x)[2]
∇f3(x) = value_and_gradient(f, AutoZygote(), x)[2]
∇f4(x) = value_and_gradient(f, AutoEnzyme(), x)[2]
x = [3.0,0]
∇fx = ∇f1(x)

scatter!([x[1]], [x[2]], label="x",color=:red)  # minimum
plot!([x[1],x[1]+∇fx[1]],[x[2],x[2]+∇fx[2]],arrow=true,color=:red, linewidth=2,
       label="∇f(x)")

println("Analytical: $(∇f1(x))")
println("ForwardDiff: $(∇f2(x))")
println("Zygote: $(∇f3(x))")
println("Enzyme: $(∇f4(x))")
