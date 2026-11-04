using CSV, DataFrames

train = CSV.read("train.csv", DataFrame)
test = CSV.read("test.csv", DataFrame)

# TODO: replace with your model
lo, hi = minimum(train.price), maximum(train.price)
predictions = lo .+ (hi - lo) .* rand(nrow(test))

CSV.write("test-predictions.csv", DataFrame(id = test.id, price = predictions))
