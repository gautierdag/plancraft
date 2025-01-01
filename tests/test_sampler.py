from plancraft.environment.sampler import (
    sample_distractors,
    assign_to_slots,
    construct_example,
)


def test_assign_to_slots():
    inventory = {
        "coal_block": 1,
        "jungle_planks": 1,
        "stripped_oak_wood": 1,
        "gray_carpet": 39,
        "ice": 6,
        "cracked_nether_bricks": 28,
        "peony": 34,
    }
    inventory_dict = assign_to_slots(inventory)
    assert len(inventory_dict) == 7
    assert sum([x["quantity"] for x in inventory_dict.values()]) == 110

    inventory = {
        "red_dye": 3,
        "cornflower": 3,
        "string": 24,
        "birch_planks": 1,
        "oak_log": 1,
    }
    inventory_dict = assign_to_slots(inventory)
    assert len(inventory_dict) == 5
    assert sum([x["quantity"] for x in inventory_dict.values()]) == 32

    inventory = {
        "nether_quartz_ore": 24,
        "yellow_bed": 1,
        "red_sandstone_slab": 5,
    }
    inventory_dict = assign_to_slots(inventory)
    assert len(inventory_dict) == 3
    assert sum([x["quantity"] for x in inventory_dict.values()]) == 30


def test_sample_distractors():
    distractors = sample_distractors(num_distractors=16)
    assert len(distractors) == 16

    distractors = sample_distractors(exclude_set=set(["apple", "diamond"]))
    assert len(distractors) == 16
    assert "apple" not in distractors
    assert "diamond" not in distractors


def test_construct_example():
    example = construct_example(target="torch", num_distractors=4)
    assert not example["impossible"]


def test_construct_example_impossible():
    example = construct_example(target="torch", num_distractors=4, impossible=True)
    assert example["impossible"]
