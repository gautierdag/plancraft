# %% [markdown]
# ### Download outputs from wandb

# %%
import wandb
from tqdm import tqdm

project = "plancraft"
entity = "itl"

api = wandb.Api()
runs = api.runs(f"{entity}/{project}")

# download all
for run in tqdm(runs):
    if "test_small" not in run.name:
        continue
    for file in run.files():
        if (
            file.name.startswith("outputs/")
            and file.name.endswith(".json")
            and "/test.small/" in file.name
        ):
            file.download(exist_ok=True)

# %% [markdown]
# ### Collate outputs into single results

# %%
import glob
import json
import os

# load from local
task_results = glob.glob("../outputs/*/test.small/*/*.json")
results = []

for task_result in task_results:
    example_id = task_result.split("/")[-1]
    seed = task_result.split("/")[-2]
    split = task_result.split("/")[-3]
    run_name = task_result.split("/")[-4]
    try:
        with open(task_result) as f:
            result = json.load(f)
    except:
        print(f"Failed to load {task_result}")
        # remove the file if it failed to load
        os.remove(task_result)
        continue
    result["name"] = run_name
    result["split"] = split
    result["seed"] = seed

    results.append(result)

# %%
with open("data/test.small.json", "r") as f:
    test_small = json.load(f)
    test_id_set = set([x["id"] for x in test_small])

# %%
example_id_to_complexity_bin = {v["id"]: int(v["complexity_bin"]) for v in test_small}

# %%
import pandas as pd


df = pd.DataFrame(results)
# df.model_trace.apply(pd.Series)
df = pd.concat([df, df.model_trace.apply(pd.Series)], axis=1).drop(
    "model_trace", axis=1
)
# ensure only ids in the test.small are included
df = df[df["example_id"].isin(test_id_set)]

df["name"] = df["name"].str.replace("Meta-Llama-3.1-8B-Instruct", "Llama_8B")
df["name"] = df["name"].str.replace("Llama-3.3-70B-Instruct", "Llama_70B")
df["name"] = df["name"].str.replace("oa-llama3-r64-a32", "Llama_8B_FT")
df["tools"] = df["name"].str.split("_").str[-1]
df["mode"] = df["name"].str.split("_").str[0]
df["modality"] = df["name"].str.split("_").str[1]
df["use_fasterrcnn"] = df["name"].str.contains("fasterrcnn")
df["model"] = (
    df["name"].str.split("_").str[2:-1].str.join("_").str.replace("fasterrcnn_", "")
)
# if oracle in name then it is an oracle model
df.loc[df["name"].str.contains("oracle"), "model"] = "oracle"
df.loc[df["name"].str.contains("oracle"), "tools"] = "m|s"

print(df["tools"].unique())
print(df["mode"].unique())
print(df["modality"].unique())
print(df["model"].unique())
print(len(df))

# %%
# df[(df.name == "act_text_Llama_8B_m|s|t|se|i") & (df.seed == "1")].tokens_used.values


# %%
# select only the full runs
full_impossible = df.groupby(["name", "seed"]).filter(
    lambda x: len(x) == len(test_small)
)
full_non_impossible = df.groupby(["name", "seed"]).filter(
    lambda x: len(x) == len([x for x in test_small if not x["impossible"]])
)
df = pd.concat([full_impossible, full_non_impossible])
print(len(df))


# %%
def get_impossible(x):
    # Whether or not the last element in the dialogue history is impossible
    if len(x) == 0:
        return False
    last_element = x[-1]["content"]
    if isinstance(last_element, list):
        last_element = last_element[-1]["text"]
    return last_element.startswith("impossible:")


def multimodal_dialogue(x):
    # Multimodal dialogue wraps content response in a list [{type: "text", text: "response"}]
    if len(x) == 0:
        return False
    if isinstance(x[-1]["content"], list):
        return True
    return False


df["multimodal_dialogue"] = df["dialogue_history"].apply(multimodal_dialogue)
df["impossible_emitted"] = df["dialogue_history"].apply(get_impossible)
df["complexity_bin"] = df["example_id"].apply(lambda x: example_id_to_complexity_bin[x])

complexity_bins_names = {
    0: "easy",
    1: "easy",
    2: "medium",
    3: "hard",
    4: "hard",
    5: "other",
}
df["original_complexity_bin"] = df["complexity_bin"]
df["complexity_bin"] = df["complexity_bin"].map(complexity_bins_names)


# Count actions
def count_actions(dialogue):
    actions = {
        "move": 0,
        "smelt": 0,
        "think": 0,
        "impossible": 0,
        "search": 0,
        "invalid": 0,
    }
    for message in dialogue:
        if message["role"] == "assistant":
            if isinstance(message["content"], list):
                for content in message["content"]:
                    if content["type"] != "text":
                        continue
                    action = content["text"].split(":")[0]
                    if action not in actions:
                        actions["invalid"] += 1
                    else:
                        actions[action] += 1
                    break
            else:
                action = message["content"].split(":")[0]
                if action not in actions:
                    actions["invalid"] += 1
                else:
                    actions[action] += 1
    # rename actions to action_counts
    out_dict = {f"{k}_count": v for k, v in actions.items()}
    out_dict["total_actions"] = sum(actions.values())
    return out_dict


# count number of actions for oracle
def oracle_count(action_history):
    actions_count = {
        "move_count": 0,
        "smelt_count": 0,
        "impossible_count": 0,
        "search_count": 0,
        "think_count": 0,
    }
    if len(action_history) == 0:
        actions_count["impossible_count"] = 1
        actions_count["total_actions"] = sum(actions_count.values())
        return actions_count
    for action in action_history:
        action_type = action["action_type"]
        actions_count[f"{action_type}_count"] += 1

    actions_count["total_actions"] = sum(actions_count.values())

    return actions_count


df["actions_dict"] = df["dialogue_history"].apply(count_actions)
df = pd.concat([df, df.actions_dict.apply(pd.Series)], axis=1).drop(
    "actions_dict", axis=1
)

# df["impossible_accuracy"] = df["impossible_emitted"] == df["impossible"]

# get dict mapping betwwen example_id and the oracle's total_actions
oracle_actions = (
    df[df["mode"] == "oracle"].groupby("example_id")["total_actions"].first().to_dict()
)
df["oracle_length"] = df["example_id"].map(oracle_actions)

# %%


# %%
print(df.groupby(["name"]).seed.nunique())
drop_names = ["oracle_text_fasterrcnn", "act_images_fasterrcnn_gpt-4o-mini_m|s"]

df = df[~df["name"].isin(drop_names)]

# %% [markdown]
# ### Results Table for Success

# %%
df.loc[(df["mode"] != "oracle") & df["success"], "actions_over_oracle"] = (
    df.loc[(df["mode"] != "oracle") & df["success"], "total_actions"]
    - df.loc[(df["mode"] != "oracle") & df["success"], "oracle_length"]
)

df.actions_over_oracle.describe()
# df[df["actions_over_oracle"] < 0]


# %%
overall_success_df = (
    df[
        (df["mode"] != "oracle")
        & (df["complexity_bin"] != "other")
        & (df["mode"] == "act")
        & (df["tools"] == "m|s|t|se|i")
    ]
    .groupby(["modality", "model", "mode", "tools", "use_fasterrcnn"])
    .success.mean()
    .reset_index()
)

overall_success_df

# %%


# overall_success_df

# (df["impossible"] == (df["complexity_bin"] == "other"))

tools = "m|s|t|se|i"
model = "Llama_70B"

number_of_incorrect_impossible = len(
    df[
        (df["tools"] == tools)
        & (df["impossible_count"] > 0)
        & (df["complexity_bin"] != "other")
        & (df["model"] == model)
    ]
)

number_of_total_incorrect = len(
    df[
        (df["tools"] == tools)
        & (~df["success"])
        & (df["complexity_bin"] != "other")
        & (df["model"] == model)
    ]
)
number_of_total_correct = len(
    df[
        (df["tools"] == tools)
        & (df["success"])
        & (df["complexity_bin"] != "other")
        & (df["model"] == model)
    ]
)
print(f"Number of incorrect impossible: {number_of_incorrect_impossible}")
print(f"Number of total incorrect: {number_of_total_incorrect}")
print(f"Number of total correct: {number_of_total_correct}")
print(
    f"Accuracy: {number_of_total_correct / (number_of_total_incorrect+number_of_total_correct)}"
)


df[
    (df["tools"] == tools)
    & (df["impossible_count"] > 0)
    & (df["complexity_bin"] != "other")
    & (df["model"] == model)
].dialogue_history.iloc[4]

# %%
# calculate f1 score for impossible
from sklearn.metrics import f1_score

df["use_fasterrcnn"] = df["use_fasterrcnn"].astype(str)

df_pivot = (
    df[df["mode"] != "oracle"]
    # .groupby(["modality", "model", "mode", "tools", "recipe_type"])
    .groupby(["modality", "model", "mode", "tools", "use_fasterrcnn", "complexity_bin"])
    .success.mean()
    .unstack(level=-1)
    .reset_index()
)

# concat with grouped by counts of actions
df_pivot = pd.merge(
    df_pivot,
    df[
        (df["mode"] != "oracle")
        & (df["complexity_bin"] != "other")
        & (df["mode"] == "act")
        # & (df["modality"] == "symb")
    ]
    .groupby(["modality", "model", "mode", "tools", "use_fasterrcnn"])
    .agg(
        {
            "total_actions": "mean",
            "move_count": "mean",
            "smelt_count": "mean",
            "think_count": "mean",
            "impossible_count": "mean",
            "search_count": "mean",
            "invalid_count": "mean",
            "actions_over_oracle": "mean",
            "tokens_used": "mean",
        }
    )
    .reset_index(),
    on=["modality", "model", "mode", "tools", "use_fasterrcnn"],
    how="left",
)

# add f1 score for impossible
df_pivot = pd.merge(
    df_pivot,
    df[df["tools"].str.contains("i")]
    .groupby(["modality", "model", "mode", "tools", "use_fasterrcnn"])
    .apply(
        lambda x: f1_score(x["impossible"], x["impossible_emitted"]),
        include_groups=False,
    )
    .reset_index(),
    on=["modality", "model", "mode", "tools", "use_fasterrcnn"],
    how="left",
)
df_pivot.rename(columns={0: "impossible f1"}, inplace=True)


# overall success
overall_success_df = (
    df[
        (df["mode"] != "oracle")
        & (df["complexity_bin"] != "other")
        & (df["mode"] == "act")
        # & (df["modality"] == "symb")
    ]
    .groupby(["modality", "model", "mode", "tools", "use_fasterrcnn"])
    .success.mean()
    .reset_index()
)


# # add into the pivot table
df_pivot = pd.merge(
    df_pivot,
    overall_success_df,
    on=["modality", "model", "mode", "tools", "use_fasterrcnn"],
    how="left",
)

# rename success to overall
df_pivot.rename(columns={"success": "overall"}, inplace=True)
# rename models to be more human readable
df_pivot["model"] = df_pivot["model"].str.replace("_", " ")

# group by modality then tools then model and sort by overall
df_pivot = df_pivot.sort_values(
    ["modality", "tools", "model", "overall"], ascending=[True, True, False, False]
)

# Replace NaN with "-"
df_pivot["impossible f1"] = df_pivot["impossible f1"].fillna("-")

# Find the maximum value in the "overall" column
max_overall = df_pivot["overall"].max()


# Apply the bolding to the 'overall' column
def bold_max(s):
    return "\\textbf{%s}" % s if s == f"{max_overall:.2f}" else s


# Apply the function to the 'overall' column
df_pivot["overall"] = df_pivot["overall"].apply(lambda x: bold_max(f"{x:.2f}"))

df_pivot = df_pivot[df_pivot["mode"] == "act"]

# %% [markdown]
# ### Format

# %%
tools_to_acronym = {
    "m|s": "M S",
    "m|s|t": "M S T",
    "m|s|t|se": "M S T SE",
    "m|s|t|se|i": "M S T SE I",
}

df_pivot["tools"] = df_pivot["tools"].map(tools_to_acronym)

# title case column names
df_pivot.columns = [x.title().replace("_", " ") for x in df_pivot.columns]

# Columns to underline the max values
columns_to_format = ["Easy", "Medium", "Hard"]
# Group by 'Tools' and format max values
for col in columns_to_format:
    max_indices = df_pivot.groupby("Tools")[col].transform("max") == df_pivot[col]
    df_pivot.loc[max_indices, col] = df_pivot.loc[max_indices, col].apply(
        lambda x: f"\\underline{{{x:.2f}}}"
    )
df_pivot["Think Count"] = df_pivot["Think Count"].replace(0, "-")
df_pivot["Search Count"] = df_pivot["Search Count"].replace(0, "-")
df_pivot["Impossible Count"] = df_pivot["Impossible Count"].replace(0, "-")


def format_thousands(x):
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    elif x >= 1_000:
        return f"{x/1_000:.1f}k"
    else:
        return f"{x}"


df_pivot["Tokens Used"] = df_pivot["Tokens Used"].apply(format_thousands)

# rename columns to remove "Count"
df_pivot = df_pivot.rename(
    columns={
        "Move Count": "Move",
        "Smelt Count": "Smelt",
        "Think Count": "Think",
        "Impossible Count": "Impossible",
        "Search Count": "Search",
        "Invalid Count": "Invalid",
        "Actions Over Oracle": "AE",  # Action efficiency
        "Total Actions": "Avg. Plan Length",
    },
)

print(df_pivot["Model"].unique())
model_order = [
    "Llama 8B",
    "Llama 70B",
    "gpt-4o-mini",
    "Llama 8B FT",
    "oam-llama3-r64-a32",
    "mrcnn gpt-4o-mini",
]

# Convert the Model column to a categorical type with the specified order
df_pivot["Model"] = pd.Categorical(
    df_pivot["Model"], categories=model_order, ordered=True
)

# Sort by Tools (ascending) and Model (following the categorical order)
df_pivot = df_pivot.sort_values(
    ["Tools", "Model"],
    ascending=[True, True],  # True for Tools, categorical handles Model order
)

# %%
# df_pivot["Use Fasterrcnn"] = df_pivot["Use Fasterrcnn"] == "True"
df_pivot[(df_pivot["Modality"] == "text") & ~(df_pivot["Use Fasterrcnn"])]


# %%
text_table = df_pivot[(df_pivot["Modality"] == "text") & ~(df_pivot["Use Fasterrcnn"])]
print(
    text_table[
        [
            "Tools",
            "Model",
            "Easy",
            "Medium",
            "Hard",
            "Overall",
            "Think",
            "Search",
            "Impossible",
            "Impossible F1",
            "Avg. Plan Length",
            "AE",
            "Tokens Used",
        ]
    ].to_latex(index=False, float_format="%.2f", escape=False)
)
# text_table

# %%
real_table = df_pivot[
    ((df_pivot["Modality"] != "text") | df_pivot["Use Fasterrcnn"])
    & (df_pivot["Tools"] == "M S")
]

print(
    real_table[
        [
            "Modality",
            "Model",
            "Overall",
            "Avg. Plan Length",
            "AE",
            "Tokens Used",
        ]
    ].to_latex(index=False, float_format="%.2f", escape=False)
)

# %% [markdown]
# ### Results Table for Plan Length

# %%
id_to_steps = (
    df[df["mode"] == "oracle"][["example_id", "number_of_steps"]]
    .set_index("example_id")
    .to_dict()["number_of_steps"]
)
# calculate steps diff between oracle and model
df["oracle_steps"] = df["example_id"].map(id_to_steps)
df["steps_diff"] = df["number_of_steps"] - df["oracle_steps"]
df.loc[~df["success"], "steps_diff"] = 10

# %%
df_pivot = (
    df[df["mode"] != "oracle"]
    .groupby(["modality", "model", "mode", "tools", "recipe_type"])
    .steps_diff.mean()
    .unstack(level=-1)
    .reset_index()
)
df_pivot["overall"] = df_pivot[["mixed", "shaped", "shapeless", "smelting"]].mean(
    axis=1
)
df_pivot["model"] = df_pivot["model"].str.replace("_", " ")

df_pivot = df_pivot.sort_values(
    ["modality", "tools", "model", "overall"], ascending=[True, True, False, False]
)

# bolden min values for overall column
min_overall = df_pivot["overall"].min()


def bold_min(s):
    return "\\textbf{%s}" % s if s == f"{min_overall:.2f}" else s


df_pivot["overall"] = df_pivot["overall"].apply(lambda x: bold_min(f"{x:.2f}"))

# in impossible column, if 10 then replace with "-"
df_pivot["impossible"] = df_pivot["impossible"].apply(lambda x: "-" if x == 10 else x)

print(
    df_pivot[
        [
            "mode",
            "tools",
            "model",
            "mixed",
            "shaped",
            "shapeless",
            "smelting",
            "overall",
            "impossible",
        ]
    ].to_latex(index=False, float_format="%.2f")
)
