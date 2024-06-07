from plancraft.environments.env_real import RealPlancraft
import imageio


starting_inv = [
    {"type": "diorite", "slot": 27, "quantity": 1},
    {"type": "cobblestone", "slot": 39, "quantity": 1},
]

env = RealPlancraft(
    inventory=starting_inv,
    symbolic_action_space=True,
    symbolic_observation_space=True,
    resolution=[512, 512],
)

actions = [
    env.action_space.no_op(),
    {"inventory_command": [27, 4, 1]},
    {"inventory_command": [39, 5, 1]},
]
images = []
for action in actions:
    obs, _, done, _ = env.step(action)
    images.append(obs["pov"])

second_inv = [
    {"type": "iron_ore", "slot": 45, "quantity": 1},
    {"type": "cobblestone", "slot": 39, "quantity": 1},
]
new_actions = [
    env.action_space.no_op(),
    {"smelt": [45, 44, 1]},
]
env.fast_reset(new_inventory=second_inv)

for action in new_actions:
    obs, _, done, _ = env.step(action)
    images.append(obs["pov"])


third_inv = [
    {"slot": 13, "type": "birch_log", "quantity": 1},
    {"slot": 36, "type": "birch_sapling", "quantity": 60},
    {"slot": 35, "type": "magma_cube_spawn_egg", "quantity": 13},
    {"slot": 31, "type": "prismarine_slab", "quantity": 10},
    {"slot": 25, "type": "painting", "quantity": 58},
    {"slot": 39, "type": "mushroom_stem", "quantity": 49},
    {"slot": 43, "type": "quartz_slab", "quantity": 17},
    {"slot": 18, "type": "light_gray_banner", "quantity": 6},
    {"slot": 23, "type": "iron_nugget", "quantity": 18},
    {"slot": 16, "type": "firework_rocket", "quantity": 32},
    {"slot": 21, "type": "villager_spawn_egg", "quantity": 49},
    {"slot": 28, "type": "melon_seeds", "quantity": 57},
    {"slot": 30, "type": "nether_quartz_ore", "quantity": 4},
    {"slot": 15, "type": "respawn_anchor", "quantity": 43},
    {"slot": 27, "type": "polished_diorite", "quantity": 36},
    {"slot": 12, "type": "polished_blackstone_button", "quantity": 53},
    {"slot": 40, "type": "netherite_axe", "quantity": 1},
]
new_actions = [
    env.action_space.no_op(),
]

env.fast_reset(new_inventory=third_inv)

for action in new_actions:
    obs, _, done, _ = env.step(action)
    images.append(obs["pov"])

# save each image for each environment
for i, img in enumerate(images):
    imageio.imsave(f"demo_env_{i}.png", img)

env.close()
