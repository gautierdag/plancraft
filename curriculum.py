# Adapt the train/test/val data to the curriculum learning
# By building a topological sort of all the recipes and processing them in order
# Some key recipes are missing from the curriculum, so also measure the # of unseen components (recipe will not have been seen within split) for each trajectory

import os
import json
import pandas as pd
from plancraft.environment.recipes import RECIPES
import networkx as nx
from pathlib import Path
from loguru import logger


# open plancraft/data/train.json
plancraft_data_path = Path("plancraft/data")
assert plancraft_data_path.exists(), "plancraft/data directory does not exist"
with open(plancraft_data_path / "train.json", "r") as f:
    # load the json data
    train_data = json.load(f)
with open(plancraft_data_path / "test.json", "r") as f:
    # load the json data
    test_data = json.load(f)
with open(plancraft_data_path / "val.json", "r") as f:
    # load the json data
    valid_data = json.load(f)


# convert the json data to a pandas dataframe
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
val_df = pd.DataFrame(valid_data)

RECIPE_GRAPH = nx.DiGraph()

for item, recipes in RECIPES.items():
    for recipe in recipes:
        RECIPE_GRAPH.add_node(recipe.result.item)
        for ingredient in recipe.inputs:
            RECIPE_GRAPH.add_node(ingredient)
            RECIPE_GRAPH.add_edge(ingredient, recipe.result.item)

# Detect and break cycles
for cycle in nx.simple_cycles(RECIPE_GRAPH):
    if len(cycle) > 1:
        # Remove one edge in the cycle (arbitrary break)
        u, v = cycle[0], cycle[1]
        if RECIPE_GRAPH.has_edge(u, v):
            RECIPE_GRAPH.remove_edge(u, v)

# Now it's safe to do a topological sort
topological_order = list(nx.topological_sort(RECIPE_GRAPH))

# order the dataframe by the topological order
train_df["topological_order"] = train_df["target"].apply(
    lambda x: topological_order.index(x)
)
test_df["topological_order"] = test_df["target"].apply(
    lambda x: topological_order.index(x)
)
val_df["topological_order"] = val_df["target"].apply(
    lambda x: topological_order.index(x)
)

# sort the dataframe by the topological order
train_df = train_df.sort_values(by="topological_order")
test_df = test_df.sort_values(by="topological_order")
val_df = val_df.sort_values(by="topological_order")


def get_unseen_components(df):
    df["unseen_component"] = 0
    for i in range(len(df)):
        if not df.iloc[i]["optimal_path"]:
            continue
        for p in df.iloc[i]["optimal_path"]:
            if p == df.iloc[i]["target"]:
                continue
            else:
                # check if the recipe is in the previously seen recipes (index <= i)
                if p not in df.iloc[:i]["target"].values:
                    df.at[i, "unseen_component"] += 1
    return df


os.makedirs("plancraft/data", exist_ok=True)

train_df = get_unseen_components(train_df)
train_df.reset_index(drop=True, inplace=True)
train_df["curriculum_order"] = train_df.index.astype(int)

coverage = (train_df["unseen_component"] == 0).sum() / len(train_df)
logger.info(f"Curriculum coverage: {coverage:.2%}")

train_df.to_json("plancraft/data/train.curriculum.json", orient="records", lines=True)

test_df = get_unseen_components(test_df)
test_df.reset_index(drop=True, inplace=True)
test_df["curriculum_order"] = test_df.index.astype(int)

coverage = (test_df["unseen_component"] == 0).sum() / len(test_df)
logger.info(f"Curriculum coverage: {coverage:.2%}")

test_df.to_json("plancraft/data/test.curriculum.json", orient="records", lines=True)

val_df = get_unseen_components(val_df)
val_df.reset_index(drop=True, inplace=True)
val_df["curriculum_order"] = val_df.index.astype(int)

coverage = (val_df["unseen_component"] == 0).sum() / len(val_df)
logger.info(f"Curriculum coverage: {coverage:.2%}")

val_df.to_json("plancraft/data/val.curriculum.json", orient="records", lines=True)
