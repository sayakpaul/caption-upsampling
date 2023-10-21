from pprint import pprint
from typing import Dict, List, Tuple

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import pipeline


def load_chat_pipeline():
    """Loads the HuggingFaceH4/zephyr-7b-alpha model and wraps into a handy text-generation pipeline."""
    pipe = pipeline(
        "text-generation",
        model="HuggingFaceH4/zephyr-7b-alpha",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return pipe


def get_messages_for_chat() -> Tuple[Dict, List[Dict]]:
    """
    Prepares the system and user-assistant style messages for inference.

    Example messages come from the DALL-E 3 technical report:
    https://cdn.openai.com/papers/dall-e-3.pdf.
    """
    system_message = {
        "role": "system",
        "content": """You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say in square brackets. For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" will trigger your partner bot to output an image of a forest morning, as described. You will be prompted by people looking to create detailed, amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.

    There are a few rules to follow:

    - You will only ever output a single image description per user request.
    - Sometimes the user will request that you modify previous captions. In this case, you should refer to your previous conversations with the user and make the modifications requested.
    - When modifications are requested, you should not simply make the description longer. You should refactor the entire description to integrate the suggestions.
    - Other times the user will not want modifications, but instead want a new image. In this case, you should ignore your previous conversation with the user."
    - Image descriptions must be between 15-80 words. Extra words will be ignored.
    """,
    }

    rest_of_the_message = [
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input: 'a man holding a sword'",
        },
        {
            "role": "assistant",
            "content": "a pale figure with long white hair stands in the center of a dark forest, holding a sword high above his head. the blade glows with a blue light , casting a soft glow on the trees and bushes surrounding him.",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'make the light red'",
        },
        {
            "role": "assistant",
            "content": "a pale figure with long white hair stands in the center of a dark forest, holding a sword high above his head. the blade glows with a red light, casting a warm glow on the trees and bushes surrounding him.",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'draw a frog playing dominoes'",
        },
        {
            "role": "assistant",
            "content": "a frog sits on a worn table playing a game of dominoes with an elderly raccoon. the table is covered in a green cloth, and the frog is wearing a jacket and a pair of jeans. The scene is set in a forest, with a large tree in the background.",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : '{drawbench_prompt}'",
        },
    ]
    return system_message, rest_of_the_message


def make_final_message(
    system_message: Dict[str, str],
    rest_of_the_message: List[Dict[str, str]],
    debug=False,
):
    """Prepares the final message for inference."""
    final_message = [system_message]
    final_message.extend(rest_of_the_message)
    if debug:
        pprint(final_message)
    return final_message


def upsample_caption(pipeline, message):
    """Performs inference on a single DrawBench prompt."""
    prompt = pipeline.tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return outputs


def prepare_assistant_reply(assistant_output):
    """Prepares the assistant reply which will be considered as the upsampled caption."""
    output = assistant_output[0]["generated_text"]
    parts = output.rsplit("<|assistant|>", 1)
    assistant_reply = parts[1].strip() if len(parts) > 1 else None
    return assistant_reply


def main():
    print("Loading dataset and pipeline...")
    drawbench = load_dataset("sayakpaul/drawbench", split="train")
    pipeline = load_chat_pipeline()

    print("Upsampling captions...")
    upsampled_captions = []
    for i in range(len(drawbench)):
        system_message, rest_of_the_message = get_messages_for_chat()
        updated_prompt = rest_of_the_message[-1]["content"].format(
            drawbench_prompt=drawbench[i]["Prompts"]
        )
        rest_of_the_message[-1]["content"] = updated_prompt
        final_message = make_final_message(
            system_message, rest_of_the_message, debug=False
        )

        outputs = upsample_caption(pipeline, final_message)
        upsampled_caption = prepare_assistant_reply(outputs)
        upsampled_captions.append(upsampled_caption)

    print("Upsampling done, pushing to the Hub...")
    data_dict = {
        "Prompt": list(drawbench["Prompts"]),
        "Upsampled Prompt": upsampled_captions,
        "Category": list(drawbench["Category"]),
    }
    dataset = Dataset.from_dict(data_dict)
    dataset.push_to_hub("drawbench-upsampled-zephyr-7b-alpha")


if __name__ == "__main__":
    main()
