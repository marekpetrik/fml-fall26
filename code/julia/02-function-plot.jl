using Plots

f(x) = exp(-2*x)
∂f(x) = -2*exp(-2*x)
∂²f(x) = 4*exp(-2*x)

x₀ = -4

f₁(x) = f(x₀) + (x - x₀) * ∂f(x₀)
f₂(x) = f(x₀) + (x - x₀) * ∂f(x₀) + 0.5 * (x - x₀)^2 * ∂²f(x₀)

δ = 4

plot(f, xlabel="x", label="f", color="black", width=2,
     xlim=[x₀ - δ/2, x₀ + δ/2])
plot!(f₁, label="∂f", color="blue", linestyle=:dash, width=2)
plot!(f₂, label="∂²f", color="green", linestyle=:dot, width=2)
