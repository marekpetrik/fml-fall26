using Plots
using LinearAlgebra
using Printf

### Choose vectors and variables

e₁ = [1 1]
e₁ = e₁ / norm(e₁)
λ₁ = 0.2
e₂ = [1 -1]
e₂ = e₂ / norm(e₂)
λ₂ = 1
@assert abs(e₁ ⋅ e₂) ≤ 1e-5    # check that they are orthogonal

println("Start with a sum of squares: ")
@printf("%01.1f (%01.1f x₁ + %01.1f x₂)²  + %01.1f (%01.1f x₁ +%01.1f x₂)²", λ₁, e₁..., λ₂, e₂...)


### Construct quadratic function

Q = hcat(e₁', e₂')
Λ = diagm([λ₁, λ₂])
A = Q * Λ * Q'

println("Turn to a quadratic function: ")
@printf("%01.1f x₁² + %01.1f x₁x₂ + %01.1f x₂²", A[1,1], A[1,2]+A[2,1], A[2,2])

### Recover the sum of quadratics

(λ, Q) = eigen(A)

println("Start with a sum of squares: ")
@printf("%01.1f (%01.1f x₁ + %01.1f x₂)²  + %01.1f (%01.1f x₁ +%01.1f x₂)²",
        λ[1], Q[:,1]..., λ[2], Q[:,2]...)

### Plot the quadratic function

x₁ = range(-3, 3, 100)
x₂ = range(-3, 3, 100)
f(x₁,x₂) = 1/2 * [x₁,x₂]' * A * [x₁,x₂]
z = f.(x₁', x₂)
contour(x₁,x₂,z, xlabel="x₁", ylabel="x₂", aspect_ratio = :equal)

plot!([0,2*λ₁*e₁[1]],[0,2*λ₁*e₁[2]],arrow=true,color=:red, linewidth=2,label="e₁")
plot!([0,2*λ₂*e₂[1]],[0,2*λ₂*e₂[2]],arrow=true,color=:blue,linewidth=2,label="e₂")

