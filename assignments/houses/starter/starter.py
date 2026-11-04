import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# TODO: replace with your model
lo, hi = train["price"].min(), train["price"].max()
predictions = np.random.uniform(lo, hi, len(test))

pd.DataFrame({"id": test["id"], "price": predictions}).to_csv("test-predictions.csv", index=False)
