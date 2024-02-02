# import hydra

# import torch
# from models.mineclip.model import MineCLIP
# from models.goal_model import get_goal_model


# @hydra.main(config_path="configs", config_name="base", version_base=None)
# def main(cfg):
#     print(cfg)
#     # Test MineCLIP
#     clip_path = cfg["pretrains"]["clip_path"]
#     clip_model = MineCLIP()
#     clip_model.load_ckpt(clip_path, strict=True)
#     clip_model.eval()
#     goal = "sheep"
#     with torch.no_grad():
#         clip_model.encode_text([goal]).cpu().numpy()

#     # Test GoalModel (ImpalaGoalCNN)
#     action_space = [3, 3, 4, 11, 11, 8, 1, 1]
#     goal_model = get_goal_model(cfg, action_space)
#     print(goal_model)


# if __name__ == "__main__":
#     main()
