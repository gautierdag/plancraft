from dotenv import load_dotenv

from plancraft.config import EvalConfig
from plancraft.environments.actions import (
    NoOp,
    SymbolicAction,
)
from plancraft.models.act import ActModel
from plancraft.models.utils import (
    convert_observation_to_message,
    parse_content_response,
)

load_dotenv()


class ReactModel(ActModel):
    """
    Model that does action with interleaved thinking step
    """

    def __init__(self, cfg: EvalConfig):
        super().__init__(cfg)
        self.max_invalid_actions = 3

    def step(self, observation: dict) -> SymbolicAction:
        # override the step method in ActModel to force thinking step

        self.history.add_observation_to_history(observation)
        observation_message = convert_observation_to_message(
            observation,
            objective=self.history.objective,
            bbox_model=self.bbox_model,
            oam_model="oam" in self.llm.model_name,
            use_text_inventory=self.use_text_inventory,
            use_multimodal_content_format=self.use_multimodal_content_format,
            use_images=self.use_images,
        )
        # add observation to history
        self.history.add_message_to_history(content=observation_message, role="user")

        i = 0
        while i < self.max_invalid_actions:
            message_window, image_window = self.llm.prepare_messages(
                history=self.history,
                max_messages_window=self.max_messages_window,
                system_prompt=self.system_prompt,
                prompt_images=self.prompt_images,
            )
            think_messages, think_token_used = self.llm.generate_unconstrained(
                batch_messages=[message_window],
                images=[image_window],
                start_messages_generation="think:",
            )
            self.history.tokens_used += think_token_used
            think_message = "think: " + think_messages[0].split("\n")[0].strip()
            self.history.add_message_to_history(content=think_message, role="assistant")

            # retrieve new message window (with thinking prompt)
            message_window, image_window = self.llm.prepare_messages(
                history=self.history,
                max_messages_window=self.max_messages_window,
                system_prompt=self.system_prompt,
                prompt_images=self.prompt_images,
            )
            action_messages, action_token_used = self.llm.generate_unconstrained(
                batch_messages=[message_window],
                images=[image_window],
                start_messages_generation="",
            )
            self.history.tokens_used += action_token_used

            action_message = action_messages[0].split("\n")[0].strip()

            self.history.add_message_to_history(
                content=action_message, role="assistant"
            )

            response = parse_content_response(
                action_message, valid_actions=self.valid_actions
            )
            if not isinstance(response, str):
                # valid action
                self.history.add_action_to_history(response)
                return response

            self.history.add_message_to_history(
                content=response,
            )
            i += 1

        # default move action
        return NoOp()
