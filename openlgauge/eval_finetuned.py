import argparse
import logging
import os
import re
from typing import Optional

from dataclasses import dataclass, field
import json
from jinja2 import Template
from unsloth import FastLanguageModel
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


SCORE_PATTERN = re.compile(r'[oO]verall score:\s*(\d+)')

@dataclass
class RetryConfig:
    max_retries: int
    temperature: float
    top_p: float
    do_sample: bool


@dataclass
class ModelConfig:
    max_seq_length: int
    max_new_tokens: int
    temperature: float
    do_sample: bool
    device: str
    retry_config: RetryConfig
    top_p: Optional[float] = None

    @classmethod
    def from_json(cls, json_path: str) -> "ModelConfig":
        with open(json_path, 'r') as f:
            config_dict = json.load(f)

        if 'retry_config' in config_dict:
            retry_config_dict = config_dict['retry_config']
            config_dict['retry_config'] = RetryConfig(**retry_config_dict)

        return cls(**config_dict)


def load_model(model_path: str, config: ModelConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load the model and tokenizer.
    
    Args:
        model_path: Path to the model
        config: Model configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=config.max_seq_length,
            load_in_4bit=False
        )
        model = FastLanguageModel.for_inference(model).to(config.device)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    prompt: str,
    streamer: TextStreamer,
    retry: bool = False
):
    messages = [
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(config.device)

    if retry:
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            streamer=streamer,
            max_new_tokens=config.max_new_tokens,
            temperature=config.retry_config.temperature,
            top_p=config.retry_config.top_p,
            do_sample=config.retry_config.do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
    else:
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            streamer=streamer,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.do_sample,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main(args: argparse.Namespace):
    model_config = ModelConfig.from_json(args.config)
    model, tokenizer = load_model(args.model, model_config)
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.dataset, 'r') as f:
        dataset = json.load(f)

    with open(args.template, 'r') as f:
        template = Template(f.read())

    with open(args.aspect_config, 'r') as f:
        aspect_config = json.load(f)

    for example in dataset:
        prompt = template.render(inputs=example['inputs'], outputs=example['outputs'], **aspect_config)

        generated_text = generate_text(model, tokenizer, model_config, prompt, streamer)
        if not re.search(SCORE_PATTERN, generated_text) and args.retry:
            for attempt in range(model_config.retry_config.max_retries):
                current_text = generate_text(model, tokenizer, model_config, prompt, streamer, retry=True)
                if re.search(SCORE_PATTERN, current_text):
                    generated_text = current_text
                    break
            if attempt < model_config.retry_config.max_retries:
                logger.warning(f"\nPattern not found, attempt {attempt + 1} of {model_config.retry_config.max_retries}")
            else:
                logger.warning(f"\nWarning: Pattern not found after {model_config.retry_config.max_retries + 1} attempts. Using last generation.")

        output_path = os.path.join(args.output_dir, example['id'])

        eval_output = {
            'inputs': example['inputs'],
            'outputs': example['outputs'],
            'result': generated_text
        }
        logger.info(example['id'] + '\n' + generated_text + '\n\n')
        logger.info(f"\nSaving {example['id']} to {output_path}\n\n")
        with open(output_path, 'w') as f:
            f.write(json.dumps(eval_output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the fine-tuned model")
    parser.add_argument("--dataset", type=str, help="Path to the dataset JSON file")
    parser.add_argument("--aspect-config", type=str, help="Path to the aspect config file")
    parser.add_argument("--template", type=str, help="Path to the prompt template")
    parser.add_argument("--config", type=str, help="Path to the inference model config file")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--retry", action="store_true", help="Retry the generation if the overall score is not found")
    args = parser.parse_args()
    main(args)
