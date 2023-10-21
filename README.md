# caption-usampling

This repository implements the idea of "caption upsampling" from [DALL-E 3](https://cdn.openai.com/papers/dall-e-3.pdf) with [Zephyr-7B](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha) and gathers results with [SDXL](https://huggingface.co/papers/2307.01952).

[images](TODO)

"Caption upsampling" is the $10 term for deriving a highly descriptive caption from a short caption. Here is an example:

**Short**: A bird scaring a scarecrow
**Upsampled**: A large, vibrant bird with an impressive wingspan swoops down from the sky, letting out a piercing call as it approaches a weathered scarecrow in a sunlit field. The scarecrow, dressed in tattered clothing and a straw hat, appears to tremble, almost as if it’s coming to life in fear of the approaching bird.

This is particularly useful in the context of text-to-image generation.

## Why does this repo exist?

DALL-E 3 uses GPT-4 for upsampling the captions. This repository aims at providing an implementation with an open-source model that is capable of performing something similar but doesn't require you to pay for the usage. As such it makes use of the "zephyr-7b-alpha" model, fine-tuned from the mighty [Mistral-7B model](https://huggingface.co/mistralai/Mistral-7B-v0.1).

You can find the upsampled captions from the DrawBench (introduced in [Imagen](https://imagen.research.google/)) benchmark dataset here: [sayakpaul/drawbench](https://huggingface.co/datasets/sayakpaul/drawbench). 

Refer to the `upsample_drawbench_captions.py` script for implementation details.

## Images with and without caption upsampling

After the DrawBench prompts were "upsampled", the `generate_images.py` script was used to generate images with the regular DrawBench prompts and the upsampled ones. You can find all the images here: [sayakpaul/drawbench-sdxl](https://huggingface.co/datasets/sayakpaul/drawbench-sdxl).

## Limitations

Since SDXL uses CLIP, upsampled captions leading to more than 77 tokens will not be fully utilized. One way to remedy this would be to change the system prompt [here](https://github.com/sayakpaul/caption-upsampling/blob/c71388f39a9717c57faffcb14c0d9152c9d78657/upsample_drawbench_captions.py#L38) so that the underlying generation model is more length-aware.

This repository uses the prompt template from the DALL-E 3 technical report (Appendix C).

## Notes

The core idea of using detailed prompts to improve the quality of the generated samples has been explored before. Readers are welcome to check out the following resources in this regard:

* "Better prompt engineering" section from [this doc](https://huggingface.co/docs/diffusers/main/en/stable_diffusion#better-prompt-engineering)
* [lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus)

