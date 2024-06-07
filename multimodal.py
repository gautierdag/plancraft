from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

image_1 = Image.open("/root/plancraft/plancraft/models/few_shot_images/0.png")
image_2 = Image.open("/root/plancraft/plancraft/models/few_shot_images/1.png")

processor = AutoProcessor.from_pretrained(
    "/nfs/public/hf/models/HuggingFaceM4/idefics2-8b-chatty", local_files_only=True
)
model = AutoModelForVision2Seq.from_pretrained(
    "/nfs/public/hf/models/HuggingFaceM4/idefics2-8b-chatty",
    device_map="auto",
    local_files_only=True,
)

BAD_WORDS_IDS = processor.tokenizer(
    ["<image>", "<fake_token_around_image>"], add_special_tokens=False
).input_ids
EOS_WORDS_IDS = [processor.tokenizer.eos_token_id]

# Create inputs
messages_batch = [
    [
        {
            "role": "system",
            "content": [{"text": "Answer like a cowboy", "type": "text"}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Where is the cobblestone?",
                },
                {"type": "image"},
            ],
        },
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What items are contained in this image?"},
            ],
        }
    ],
]
images = [[image_1], [image_2]]
prompt = processor.apply_chat_template(messages_batch, add_generation_prompt=True)
inputs = processor(text=prompt, images=images, return_tensors="pt", padding=True)
inputs = {k: v.to("cuda") for k, v in inputs.items()}


# Generate
generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts[0])
print(generated_texts[1])
