# caption-upsampling

This repository implements the idea of "caption upsampling" from [DALL-E 3](https://cdn.openai.com/papers/dall-e-3.pdf) with [Zephyr-7B](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha) and gathers results with [SDXL](https://huggingface.co/papers/2307.01952).

<table>
    <tr>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/sandwich.jpg" alt="Sample Image 1"></td>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/car_sheep.jpg" alt="Sample Image 2"></td>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/owl.jpg" alt="Sample Image 3"></td>
    </tr>
    <tr>
        <td><b>A white colored sandwich.</b></td>
        <td><b>A white car and a red sheep.</b></td>
        <td><b>A side view of an owl sitting in a field.</b></td>
    </tr>
    <tr>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/upsampled_sandwich.jpg" alt="Sample Image 4"></td>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/upsampled_car_sheep.jpg" alt="Sample Image 5"></td>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/upsampled_owl.jpg" alt="Sample Image 6"></td>
    </tr>
    <tr>
        <td><b>A white-bread sandwich with delicate layers of fluffy turkey, crisp lettuce, and juicy tomatoes is placed on a wooden cutting board. The sandwich is surrounded by various condiments, including mayonnaise, mustard, and a small jar of pickles. The scene is set in a cozy kitchen, with natural light pouring in through a window.</b></td>
        <td><b>A white car is parked on the side of a road in a green meadow. In the distance, a flock of red sheep can be seen grazing. The car seems to be abandoned, and the windows are shattered. The scene is eerie, and there is an unsettling feeling in the air.</b></td>
        <td><b>A regal-looking snowy owl perches on a rocky outcropping, its feathers fluffed against the chilly wind. The bird's large, yellow eyes are fixed on a rabbit nibbling on some grass in the distance. The sun sets behind the owl, casting a warm orange glow over the landscape.</b></td>
    </tr>
</table>
<sub>Explore more samples <a href="https://huggingface.co/datasets/sayakpaul/drawbench-sdxl">here</a>. Find additional examples <a href="https://github.com/sayakpaul/caption-upsampling#additional-examples">below</a> with SDXL Refiner and Kandinsky V2.2.</sub>
<br><br>

"Caption upsampling" is the $10 term for deriving a highly descriptive caption from a short caption. Here is an example:

**Short**: _A bird scaring a scarecrow_

**Upsampled**: _A large, vibrant bird with an impressive wingspan swoops down from the sky, letting out a piercing call as it approaches a weathered scarecrow in a sunlit field. The scarecrow, dressed in tattered clothing and a straw hat, appears to tremble, almost as if it’s coming to life in fear of the approaching bird._

This is particularly useful in the context of text-to-image generation.

🌟 **Update 23/10/2023**: Got featured in this [TLDR newsletter](https://tldr.tech/ai/2023-10-23).

## Why does this repo exist?

DALL-E 3 uses GPT-4 for upsampling the captions. This repository aims at providing an implementation with an open-source model that is capable of performing something similar but doesn't require you to pay for the usage. As such it makes use of the "zephyr-7b-alpha" model, fine-tuned from the mighty [Mistral-7B model](https://huggingface.co/mistralai/Mistral-7B-v0.1).

You can find the upsampled captions from the DrawBench (introduced in [Imagen](https://imagen.research.google/)) benchmark dataset here: [sayakpaul/drawbench](https://huggingface.co/datasets/sayakpaul/drawbench). 

Refer to the `upsample_drawbench_captions.py` script for implementation details.

## Images with and without caption upsampling

After the DrawBench prompts were "upsampled", the `generate_images.py` script was used to generate images with the regular DrawBench prompts and the upsampled ones. You can find all the images here: [sayakpaul/drawbench-sdxl](https://huggingface.co/datasets/sayakpaul/drawbench-sdxl).

## Additional examples

This section presents results generated using the SDXL Refiner and Kandinsky V2.2. These were generated using the scripts from the `additional_examples` directory.

### SDXL Refiner 

<table>
    <tr>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/refiner/sandwich.jpg" alt="Sample Image 1"></td>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/refiner/car_sheep.jpg" alt="Sample Image 2"></td>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/refiner/owl.jpg" alt="Sample Image 3"></td>
    </tr>
    <tr>
        <td><b>A white colored sandwich.</b></td>
        <td><b>A white car and a red sheep.</b></td>
        <td><b>A side view of an owl sitting in a field.</b></td>
    </tr>
    <tr>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/refiner/upsampled_sandwich.jpg" alt="Sample Image 4"></td>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/refiner/upsampled_car_sheep.jpg" alt="Sample Image 5"></td>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/refiner/upsampled_owl.jpg" alt="Sample Image 6"></td>
    </tr>
    <tr>
        <td><b>A white-bread sandwich with delicate layers of fluffy turkey, crisp lettuce, and juicy tomatoes is placed on a wooden cutting board. The sandwich is surrounded by various condiments, including mayonnaise, mustard, and a small jar of pickles. The scene is set in a cozy kitchen, with natural light pouring in through a window.</b></td>
        <td><b>A white car is parked on the side of a road in a green meadow. In the distance, a flock of red sheep can be seen grazing. The car seems to be abandoned, and the windows are shattered. The scene is eerie, and there is an unsettling feeling in the air.</b></td>
        <td><b>A regal-looking snowy owl perches on a rocky outcropping, its feathers fluffed against the chilly wind. The bird's large, yellow eyes are fixed on a rabbit nibbling on some grass in the distance. The sun sets behind the owl, casting a warm orange glow over the landscape.</b></td>
    </tr>
</table>
<sub>Explore more samples <a href="https://huggingface.co/datasets/sayakpaul/drawbench-sdxl-refiner">here</a>.</sub>
<br><br>

### Kandinsky V2.2

<table>
    <tr>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/kandinsky_v22/sandwich.jpg" alt="Sample Image 1"></td>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/kandinsky_v22/car_sheep.jpg" alt="Sample Image 2"></td>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/kandinsky_v22/owl.jpg" alt="Sample Image 3"></td>
    </tr>
    <tr>
        <td><b>A white colored sandwich.</b></td>
        <td><b>A white car and a red sheep.</b></td>
        <td><b>A side view of an owl sitting in a field.</b></td>
    </tr>
    <tr>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/kandinsky_v22/upsampled_sandwich.jpg" alt="Sample Image 4"></td>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/kandinsky_v22/upsampled_car_sheep.jpg" alt="Sample Image 5"></td>
        <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/caption-upsampling/kandinsky_v22/upsampled_owl.jpg" alt="Sample Image 6"></td>
    </tr>
    <tr>
        <td><b>A white-bread sandwich with delicate layers of fluffy turkey, crisp lettuce, and juicy tomatoes is placed on a wooden cutting board. The sandwich is surrounded by various condiments, including mayonnaise, mustard, and a small jar of pickles. The scene is set in a cozy kitchen, with natural light pouring in through a window.</b></td>
        <td><b>A white car is parked on the side of a road in a green meadow. In the distance, a flock of red sheep can be seen grazing. The car seems to be abandoned, and the windows are shattered. The scene is eerie, and there is an unsettling feeling in the air.</b></td>
        <td><b>A regal-looking snowy owl perches on a rocky outcropping, its feathers fluffed against the chilly wind. The bird's large, yellow eyes are fixed on a rabbit nibbling on some grass in the distance. The sun sets behind the owl, casting a warm orange glow over the landscape.</b></td>
    </tr>
</table>
<sub>Explore more samples <a href="https://huggingface.co/datasets/sayakpaul/drawbench-kandinsky-v22">here</a>.</sub>
<br><br>

## Limitations ⛔️

1. Since SDXL uses CLIP, upsampled captions leading to more than 77 tokens will not be fully utilized. One way to remedy this would be to change the system prompt [here](https://github.com/sayakpaul/caption-upsampling/blob/c71388f39a9717c57faffcb14c0d9152c9d78657/upsample_drawbench_captions.py#L38) so that the underlying generation model is more length-aware.

   This repository uses the prompt template from the DALL-E 3 technical report (Appendix C).

2. DALL-E 3 conducts training on a recaptioned dataset where the captions were regenerated to be much more detailed using GPT-4. It then demonstrates the effectiveness of using detailed prompts during inference. However, existing works (as noted in [here](#notes)) show that it's possible to improve the generation quality of existing systems like SDXL with detailed prompts even when they weren't particularly trained on similar datasets with very detailed captions.

3. It's important to investigate the output of the language model that's producing the descriptive captions. This directly impacts the quality of the images. As mentioned above, the prompt template is the original one used in the DALL-E 3 report. However, different language models might respond differently to that template. So, figuring out which template gives the best output most of the time is crucial.

## Notes

The core idea of using detailed prompts to improve the quality of the generated samples has been explored before. Readers are welcome to check out the following resources in this regard:

* "Better prompt engineering" section from [this doc](https://huggingface.co/docs/diffusers/main/en/stable_diffusion#better-prompt-engineering)
* [lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus)

Additionally, [PixArt-Alpha](https://github.com/PixArt-alpha/PixArt-alpha) shows that fine-tuning on a dataset with highly detailed captions can lead to substantial quality improvements.
