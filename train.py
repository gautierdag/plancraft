import logging
import warnings

import hydra
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    Idefics2ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

import wandb
from plancraft.config import TrainConfig
from plancraft.train.dataset import get_dataset_and_collate

wandb.require("core")

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg):
    logger.info(cfg)
    cfg = TrainConfig(**dict(cfg))
    torch.set_float32_matmul_precision("medium")

    if cfg.training.base_model == "llama3":
        model_name = "/nfs/public/hf/models/meta-llama/Meta-Llama-3-8B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        target_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
        ]
    elif cfg.training.base_model == "llama3.1":
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        target_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
        ]
    elif cfg.training.base_model == "idefics2":
        model_name = "/nfs/public/hf/models/HuggingFaceM4/idefics2-8b-chatty"
        model = Idefics2ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        target_modules = ".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"
    else:
        raise ValueError(f"Model {cfg.training.base_model} not supported")

    train_dataset, val_dataset, collate_fn = get_dataset_and_collate(
        template_name=cfg.training.base_model,
        max_length=cfg.training.max_seq_length,
        max_message_window=cfg.training.max_message_window,
        trace_mode=cfg.training.trace_mode,
    )

    lora_config = LoraConfig(
        r=cfg.training.lora_r,
        lora_alpha=cfg.training.lora_alpha,
        lora_dropout=cfg.training.lora_dropout,
        target_modules=target_modules,
        use_dora=True,
        init_lora_weights="gaussian",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config, adapter_name="default")
    model.print_trainable_parameters()

    name = f"{cfg.training.trace_mode}-{cfg.training.base_model}-r{cfg.training.lora_r}-a{cfg.training.lora_alpha}"

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        config=cfg.model_dump(),
        name=name,
    )

    training_args = TrainingArguments(
        output_dir=f"outputs/{name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        num_train_epochs=cfg.training.num_train_epochs,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        learning_rate=cfg.training.learning_rate,
        optim="adamw_hf",
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.training.warmup_ratio,
        dataloader_num_workers=cfg.training.num_workers,
        dataloader_pin_memory=True,
        logging_dir=f"outputs/logs/{name}",
        logging_steps=1,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=False,
        bf16=True if torch.cuda.is_bf16_supported() else False,  # bf16 support check
        report_to="wandb",
    )

    # Initialize the Huggingface Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()

    # save model
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"outputs/{name}/{name}")

    if cfg.training.push_to_hub:
        merged_model.push_to_hub(name)


if __name__ == "__main__":
    main()
