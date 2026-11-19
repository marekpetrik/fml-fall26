using CSV, DataFrames, Plots
using LinearAlgebra, Statistics

cd(@__DIR__)
mkpath("figures")

### --- Load the data

df = CSV.read("climate.csv", DataFrame)
X = Matrix(df[:, r"^d\d"])       # cities × days 
m, n = size(X)
println("Data dimensions: ", (m, n))

days = 1:n
months = ([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
          ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
row(city) = findfirst(==(city), df.city)

### --- Temperature curves of a few cities

some = ["Durham", "Boston", "Anchorage", "San Francisco", "Jackson", "Las Vegas"]
plot(xticks = months, xlabel = "day of year", ylabel = "mean temperature (°C)", legend = :outerright)
for c in some
    plot!(days, X[row(c), :], label = c)
end
savefig("figures/curves.pdf")

### --- PCA: eigenvectors of the covariance matrix

μ = vec(mean(X, dims = 1))         # mean temperature curve
Xc = X .- μ'                       # centered data: rows are xᵢ - μ
S = Xc' * Xc / m                   # n × n sample covariance matrix

E = eigen(Symmetric(S))
order = sortperm(E.values, rev = true)
λ = E.values[order]                # variances along the principal directions
V = E.vectors[:, order]            # principal directions (eigenvectors) in columns

# eigenvectors are unique only up to sign; pick signs that are easy to interpret
V[:, 1] *= sign(sum(V[:, 1]))      # PC1 > 0: warmer than average
V[:, 2] *= sign(V[200, 2])         # PC2 > 0: warmer in summer (mid-July)
V[:, 3] *= sign(V[300, 3])         # PC3 > 0: warmer in the fall (late October)

# the same directions from the SVD of the centered data: Xc = U Σ Wᵀ
σ = svdvals(Xc)
println("Eigenvalues match the SVD: ", λ[1:5] ≈ σ[1:5] .^ 2 / m)
println("Nonzero eigenvalues (part 2): ", count(λ .> 1e-8 * λ[1]), " of ", n)

explained = λ / sum(λ)
for k in 1:5
    println("PC$k: ", round(100 * explained[k], digits = 2), "% of variance, ",
            round(100 * sum(explained[1:k]), digits = 2), "% cumulative")
end
for target in (0.95, 0.99)
    println("Components for $(round(Int, 100target))% of variance: ",
            findfirst(cumsum(explained) .≥ target))
end

bar(1:10, 100 * explained[1:10], label = "PC k", xlabel = "k", ylabel = "% variance explained",
    xticks = 1:10)
plot!(1:10, 100 * cumsum(explained[1:10]), marker = :circle, label = "PC 1 to k")
savefig("figures/variance.pdf")

### --- Principal directions as functions of the day of the year

plot(xticks = months, xlabel = "day of year", ylabel = "loading")
for k in 1:3
    plot!(days, V[:, k], label = "PC$k ($(round(100 * explained[k], digits = 1))%)", lw = 2)
end
hline!([0], color = :gray, label = "")
savefig("figures/eigenvectors.pdf")

# the mean curve plus and minus two standard deviations along each direction
plots = map(1:3) do k
    δ = 2 * sqrt(λ[k]) * V[:, k]
    plot(days, μ, color = :black, label = "mean", xticks = months, title = "PC$k",
         legend = k == 1 ? :bottom : false, ylabel = "°C")
    plot!(days, μ + δ, color = :red, label = "+2 sd")
    plot!(days, μ - δ, color = :blue, label = "-2 sd")
end
plot(plots..., layout = (1, 3), size = (1200, 350), margin = 4Plots.mm)
savefig("figures/perturbations.pdf")

### --- Cities differ much more in winter than in summer, so PC1 is not a constant

sd = vec(std(X, dims = 1))
println("Std. deviation across cities: January ", round(mean(sd[1:31]), digits = 1),
        " °C, July ", round(mean(sd[182:212]), digits = 1), " °C")


### --- Cities projected onto the principal components (their scores)

Z = Xc * V[:, 1:3]                 # m × 3 scores; row i is zᵢ = V₃ᵀ (xᵢ - μ)
colors = ifelse.(df.city .∈ Ref(some), :red, :blue)

function embedding(i, j)
    scatter(Z[:, i], Z[:, j], markercolor = colors, markerstrokewidth=0, markersize=5; label = "",
            xlabel = "PC$i score", ylabel = "PC$j score", size = (1000, 800))
    annotate!(Z[:, i], Z[:, j], text.("   " .* df.city, 5, :left))
end
embedding(1, 2)
savefig("figures/embedding12.pdf")
embedding(2, 3)
savefig("figures/embedding23.pdf")

### --- Reconstruction of the cities in part 1 with k principal components

println("\nRMSE (°C) of the reconstructions with k = 1, 2, 3 principal components:")
plots = map(some) do city
    i = row(city)
    plt = plot(days, X[i, :], color = :black, lw = 2, label = "actual", title = city,
               xticks = (months[1][1:2:end], months[2][1:2:end]), ylabel = "°C",
               legend = :bottom, legendfontsize = 6)
    rmse = zeros(3)
    for k in 1:3
        x̂ = μ + V[:, 1:k] * (V[:, 1:k]' * Xc[i, :])
        rmse[k] = norm(X[i, :] - x̂) / sqrt(n)
        plot!(days, x̂, label = "k = $k (RMSE $(round(rmse[k], digits = 2)) °C)")
    end
    println(rpad("  " * city, 18), round.(rmse, digits = 2))
    plt
end
plot(plots..., layout = (2, 3), size = (1200, 700), margin = 3Plots.mm)
savefig("figures/reconstruction.pdf")


### --- Standardizing the days gives summer the same weight as winter

Xs = Xc ./ sd'
v₁ = eigen(Symmetric(Xs' * Xs / m)).vectors[:, end]
v₁ *= sign(sum(v₁))
plot(days, [V[:, 1] v₁], label = ["centered" "standardized"], xticks = months,
     ylabel = "PC1 loading", lw = 2)
savefig("figures/standardized.pdf")
