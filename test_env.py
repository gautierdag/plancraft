from minedojo import MineDojoEnv
import numpy as np
import torch


rgb_list = []
env = MineDojoEnv(
    name="Plains",
    img_size=(640, 480),
    rgb_only=False,
    fast_reset=False,
    force_slow_reset_interval=10,
)
obs = env.reset()

done = False
for i in range(100):
    if done:
        continue
    act = env.action_space.no_op()
    act[0] = 1  # forward/backward
    if i % 10 == 0:
        act[2] = 1  # jump
    obs, reward, done, info = env.step(act)
    # simulate model inference
    # add to list
    rgb_list.append(obs["rgb"])
env.close()

rgbs = np.stack(rgb_list)
rgbs = torch.from_numpy(rgbs)
assert rgbs.shape == (100, 480, 640, 3)
# torchvision.io.write_video("out.mp4", rgbs, fps=20)
