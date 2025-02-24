import argparse
import json
from dataclasses import dataclass

from datasets import Dataset
from jinja2 import Template
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported


@dataclass
class LoRAConfig:
    rank: int
    alpha: int
    target_modules: list[str]
    dropout: float
    bias: str


@dataclass
class ModelConfig:
    # Model configuration
    max_seq_length: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    dataset_num_proc: int
    packing: bool

    # Training parameters
    learning_rate: float
    num_train_epochs: int
    warmup_steps: int
    weight_decay: float
    logging_steps: int
    optimizer: str
    lr_scheduler_type: str
    seed: int
    lora_config: LoRAConfig

    @classmethod
    def from_json(cls, json_path: str) -> "ModelConfig":
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        if 'lora_config' in config_dict:
            lora_dict = config_dict['lora_config']
            config_dict['lora_config'] = LoRAConfig(**lora_dict)

        return cls(**config_dict)


def create_instruction_dataset(dataset_path: str, template_path: str) -> Dataset:
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    with open(template_path, 'r') as f:
        template = Template(f.read())

    formatted_data = []
    for example in data:
        prompt = template.render(**example)
        formatted_text = f"""{prompt}

### Evaluation:
{example['evaluation']}"""
        formatted_data.append({
            "input_id": f"{example['dataset']}-{example['input_id']}", 
            "text": formatted_text
        })
    
    return Dataset.from_list(formatted_data)


def setup_model(config):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=False
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_config.rank,
        lora_alpha=config.lora_config.alpha,
        target_modules=config.lora_config.target_modules,
        lora_dropout=config.lora_config.dropout,
        bias=config.lora_config.bias,
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )
    
    return model, tokenizer


def get_training_config(args, config):
    """Configure training arguments."""
    return TrainingArguments(
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=config.logging_steps,
        optim=config.optimizer,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        output_dir=args.output_dir,
        report_to="none",
    )


def main(args):
    config = ModelConfig.from_json(args.config)

    model, tokenizer = setup_model(config)
    train_dataset = create_instruction_dataset(args.dataset, args.template)
    train_dataset.save_to_disk("train_dataset")

    print(f"Train dataset size: {len(train_dataset)}")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=config.dataset_num_proc,
        packing=config.packing,
        args=get_training_config(args, config)
    )

    trainer.train()
    model.save_pretrained(args.model_name)
    tokenizer.save_pretrained(args.model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning multi-task model")
    parser.add_argument('--dataset', type=str, help='Path to the training dataset')
    parser.add_argument('--template', type=str, help='Path to the prompt template')
    parser.add_argument('--config', type=str, help='Path to the training config file')
    parser.add_argument('--model-name', type=str, help='Name of the model')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for checkpoints')
    args = parser.parse_args()
    main(args)
