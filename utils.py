import logging
import sys

from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('prompts'))
intro_template = env.get_template("intro.j2")
intro_prompt = intro_template.render()
return_json_template = env.get_template("return_json.j2")
return_json_instructions = return_json_template.render()


def setup_logger(name: str = 'app', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def render_template(filename: str, args: dict, include_intro: bool = True, include_return_instructions: bool = True) -> str:
    """Prepare a prompt from jinja template."""

    if include_intro:
        args["intro"] = intro_prompt
    if include_return_instructions:
        args["return_json_instructions"] = return_json_instructions
    template = env.get_template(filename)
    prompt = template.render(**args)
    return prompt
