# plancraft

[![Test](https://github.com/gautierdag/plancraft/actions/workflows/test.yaml/badge.svg)](https://github.com/gautierdag/plancraft/actions/workflows/test.yaml)
![Python Version](https://img.shields.io/badge/python-3.9+-blue)
![Ruff](https://img.shields.io/badge/linter-ruff-blue)
[![PyPI Version](https://img.shields.io/pypi/v/plancraft)](https://pypi.org/project/plancraft/)

Plancraft is a minecraft environment and agent that innovates on planning LLM agents with a retriever

You can install the package by running the following command:

```bash
pip install plancraft
```

## Usage

The package provides a `PlancraftEnvironment` class that can be used to interact with the environment. Here is an example of how to use it:

```python
from plancraft.environments.env import PlancraftEnvironment


def main():
    # Create the environment with an inventory containing 10 iron ores and 23 oak logs
    env = PlancraftEnvironment(
        inventory=[dict(type="iron_ore", quantity=10, slot=10)]
        + [dict(type="oak_log", quantity=23, slot=23)],
    )
    # move one log to slot 1
    # [from, to, quantity]
    move_action = dict(
        move=[23, 2, 1]
    )
    observation = env.step(move_action)
    # observation["inventory"] contains the updated symbolic inventory
    # observation["image"] contains the updated image of the inventory

    # smelt one iron ore
    # [from, to, quantity]
    smelt_action = dict(
        smelt=[10, 11, 1]
    )
    observation = env.step(smelt_action)

    # no op
    observation = env.step()
```

Note that the environment is deterministic and stateful, so the same action will always lead to the same observation and the environment will keep track of the state of the inventory.

### Evaluator

The package also provides an `Evaluator` class that can be used to evaluate the performance of an agent on our specific dataset. Here is an example of how to use it:

```python
from plancraft.evaluator import Evaluator
from plancraft.config import EvalConfig

def main():
    # Create the config
    config = EvalConfig(...)
    # Create the evaluator
    evaluator = Evaluator(config)
    # Evaluate the agent
    evaluator.eval_all_seeds()
```

The evaluator class handles the environment loop and model interaction. The environment is created based on the configuration and the examples are loaded from the dataset. The `Evaluator` uses the dataset examples and initializes the environment with the example's inventory. It is also responsible for early stopping and verifying the target object has been craft. Finally, it also saves the results of the evaluation and the images generated during the evaluation.

## Docker

There is a docker image built to incorporate the latest code and its dependencies. I build it by running the following command:

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t gautierdag/plancraft --push .
```

The image is available on [Docker Hub](https://hub.docker.com/r/gautierdag/plancraft). Note that, unlike the package, the docker image includes everything in the repo.

### TODO

- [x] Migrate models to return actions as str
  - [x] Dummy Model
  - [x] Act Model
  - [x] Oracle Model
  - [x] Evaluator should handle tools and process the raw string
    - [x] Move `parse_content_response` to evaluator
    - [x] Move `gold_search_recipe` into environment module (search.py)
  - [x] Evaluator should handle the case where three non-env tools are used in a row -> force an observation/goal of the inventory
- [ ] History should be attached to Evaluator object. Models should track whatever they need independently
- [ ] Observations should be passed as either text or image or both. Already in the correct format
- [ ] Rerun image models with better bound box
  - [ ] Track bounding box accuracy
- [ ] Set up github pages website for repo
- [ ] Reduce size of inventory object - use dict instead of list, don't add empty quantities
