import argparse
import json
import logging
import os
from jinja2 import Template
from pathlib import Path

from ollama import chat

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_ollama(prompt: str, model: str) -> str:
    response = chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']


def run_evaluation(template: Template, aspect_config: dict, data: dict, model: str, output_dir: str):
    for example in data:
        prompt = template.render(
            inputs=example['inputs'],
            outputs=example['outputs'],
            **aspect_config
        )
        result = run_ollama(prompt, model)
        output_path = Path(output_dir) / f'{example["id"]}.json'
        eval_output = {
            'inputs': example['inputs'],
            'outputs': example['outputs'],
            'result': result
        }
        logger.info(example['id'] + '\n' + result + '\n\n')
        logger.info(f"\nSaving {example['id']} to {output_path}\n\n")
        with open(output_path, 'w') as f:
            f.write(json.dumps(eval_output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=str, help='Path to the prompt template file')
    parser.add_argument('--aspect-config', type=str, help='Path to the aspect configuration file')
    parser.add_argument('--data', type=str, help='Path to the datasets with inputs and outputs to evaluate')
    parser.add_argument('--model', type=str, help='Ollama model name')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    args = parser.parse_args()

    template_path = Path(args.template)
    aspect_config_path = Path(args.aspect_config)
    data_path = Path(args.data)

    with open(template_path, 'r') as f:
        template = Template(f.read())

    with open(aspect_config_path, 'r') as f:
        aspect_config = json.load(f)

    with open(data_path, 'r') as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    run_evaluation(template, aspect_config, data, args.model, args.output_dir)
