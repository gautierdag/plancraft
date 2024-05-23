from plancraft.environments.env_real import RealPlancraft
import imageio

envs = []
images = {}

for i, obj in enumerate(["iron_ore", "oak_log"]):
    env = RealPlancraft(
        inventory=[{"slot": i + 10, "type": obj, "quantity": i + 10}],
        symbolic_action_space=True,
        symbolic_observation_space=True,
    )
    envs.append(env)
    images[i] = []


for i in range(10):
    for j, env in enumerate(envs):
        action = env.action_space.no_op()
        # action["inventory_command"] = [i, i, 1]
        obs, _, done, _ = env.step(action)
        images[j].append(obs["pov"])


# close all environments
for env in envs:
    env.close()

# save gifs for each environment
for i, img_list in images.items():
    imageio.mimsave(f"env_{i}.gif", img_list)
