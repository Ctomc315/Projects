import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# This script expects ``df_filtered`` to be a pandas DataFrame already loaded
# with the columns specified below.  The numerical columns are cleaned and used
# to engineer additional features before training a simple decision tree model
# that ranks real estate agents.

NUMERICAL_COLUMNS = [
    "SqFt",
    "Seller Concession",
    "Concession %",
    "Original List Price",
    "True $/SqFt",
    "Sold-to-List",
    "Sold to List Price",
    "Days on Market",
    "Culmulative Days on Market",
    "Year Built",
    "True Close Price",
]


def preprocess_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dollar and percent formatted strings to numeric."""
    df = df.copy()
    df[NUMERICAL_COLUMNS] = (
        df[NUMERICAL_COLUMNS]
        .replace({r"[\$,]": "", r"%": ""}, regex=True)
        .apply(pd.to_numeric, errors="coerce")
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features used for ranking agents."""
    df = df.copy()

    # Provided features
    df["Price_per_SqFt"] = df["True Close Price"] / df["SqFt"]
    df["DOM_inv"] = df["Days on Market"].max() - df["Days on Market"]

    # Additional engineered features
    df["Concession_Ratio"] = df["Seller Concession"] / df["True Close Price"]
    df["Price_vs_List"] = df["True Close Price"] - df["Original List Price"]

    scaler = MinMaxScaler()
    to_scale = ["Price_per_SqFt", "DOM_inv", "Concession_Ratio", "Price_vs_List"]
    df[to_scale] = scaler.fit_transform(df[to_scale])

    # Simple weighted score from the original two engineered features
    weights = {
        "Price_per_SqFt": 0.50,
        "DOM_inv": 0.50,
    }
    df["Agent_Score"] = sum(df[col] * w for col, w in weights.items())
    return df


def train_decision_tree(df: pd.DataFrame) -> DecisionTreeRegressor:
    """Train a DecisionTreeRegressor to predict ``Agent_Score``."""
    features = ["Price_per_SqFt", "DOM_inv", "Concession_Ratio", "Price_vs_List"]
    X = df[features]
    y = df["Agent_Score"]

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(X_train, y_train)
    df["Predicted_Score"] = tree.predict(X)
    return tree


def rank_agents(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of agents ranked by ``Predicted_Score``."""
    ranked = df.sort_values("Predicted_Score", ascending=False)
    return ranked


def build_recommendation_map(df: pd.DataFrame) -> pd.Series:
    """Build a mapping from (City, Zip, Subdivision) to the best agent."""
    score_by_loc = (
        df.groupby([
            "City",
            "Zip Code",
            "Subdivision",
            "List Agent Full Name",
        ])[
            "Predicted_Score"
        ].mean()
    )

    top_agents = (
        score_by_loc.groupby(level=[0, 1, 2])
        .idxmax()
        .apply(lambda x: x[3])
    )
    return top_agents


def recommend_agent(
    recommendation_map: pd.Series, city: str, zip_code: str, subdivision: str
) -> str:
    """Return the recommended agent for a given location."""
    return recommendation_map.loc[(city, zip_code, subdivision)]


if __name__ == "__main__":
    # ``df_filtered`` should be defined elsewhere. The lines below illustrate
    # typical usage and assume the variable is already available.
    try:
        df_filtered = preprocess_numeric(df_filtered)
        df_filtered = engineer_features(df_filtered)
        model = train_decision_tree(df_filtered)
        ranked_agents = rank_agents(df_filtered)
        print(ranked_agents[["List Agent Full Name", "Predicted_Score"]].head())

        rec_map = build_recommendation_map(df_filtered)
        print("Recommendation map built successfully.")
    except NameError:
        print("df_filtered DataFrame is not defined.")
