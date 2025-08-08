# import pandas as pd
# from sklearn.datasets import load_iris

# def save_iris_to_csv(output_path="data/iris.csv"):
#     iris = load_iris(as_frame=True)
#     df = iris.frame
#     df.to_csv(output_path, index=False)
#     print(f"Iris dataset saved to {output_path}")

# if __name__ == "__main__":
#     save_iris_to_csv()

# src/preprocess.py

import pandas as pd
from sklearn.datasets import fetch_california_housing

def save_housing_to_csv(output_path="data/housing.csv"):
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df.to_csv(output_path, index=False)
    print(f"California Housing dataset saved to {output_path}")

if __name__ == "__main__":
    save_housing_to_csv()
