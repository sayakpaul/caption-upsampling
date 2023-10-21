import torch
from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value, load_dataset
from diffusers import DiffusionPipeline

BATCH_SIZE = 4


def main():
    print("Loading dataset...")
    drawbench = load_dataset(
        "sayakpaul/drawbench-upsampled-zephyr-7b-alpha", split="train"
    )

    print("Loading pipeline...")
    ckpt_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DiffusionPipeline.from_pretrained(ckpt_id, torch_dtype=torch.float16).to(
        "cuda"
    )
    pipe.set_progress_bar_config(disable=True)

    seed = 0
    generator = torch.manual_seed(seed)

    print("Running inference...")
    main_dict = {}
    image_path_ids = list(range(len(drawbench)))
    regular_caption_paths = list(map(lambda i: f"sdxl_{i}.png", image_path_ids))
    upsampled_caption_paths = list(
        map(lambda i: f"sdxl_upsampled_{i}.png", image_path_ids)
    )

    for i in range(0, len(drawbench), BATCH_SIZE):
        samples = drawbench[i : i + BATCH_SIZE]

        # Regular captions.
        prompts = list(samples["Prompt"])
        images = pipe(prompts, generator=generator, num_inference_steps=25).images
        for j in range(len(images)):
            images[j].save(regular_caption_paths[j])

        # Upsampled captions.
        usampled_prompts = list(samples["Upsampled Prompt"])
        images = pipe(
            usampled_prompts, generator=generator, num_inference_steps=25
        ).images
        for j in range(len(images)):
            images[j].save(upsampled_caption_paths[j])

    for i in range(drawbench):
        sample = drawbench[i]
        main_dict.update(
            {
                i: {
                    "Prompt": sample[i]["Prompt"],
                    "Image": regular_caption_paths[i],
                    "Upsampled_Prompt": sample[i]["Upsampled Prompt"],
                    "Image_With_Upsampled_Prompt": upsampled_caption_paths[i],
                }
            }
        )

    def generation_fn():
        for i in main_dict:
            entry = main_dict[i]
            yield {
                "Prompt": entry["Prompt"],
                "Image": entry["Image"],
                "Upsampled_Prompt": entry["Upsampled_Prompt"],
                "Image_With_Upsampled_Prompt": entry["Image_With_Upsampled_Prompt"],
                "model_name": ckpt_id,
                "seed": seed,
            }

    print("Preparing HF dataset...")
    ds = Dataset.from_generator(
        generation_fn,
        features=Features(
            Prompt=Value("string"),
            Image=ImageFeature(),
            Upsampled_Prompt=Value("string"),
            Image_With_Upsampled_Prompt=ImageFeature(),
            model_name=Value("string"),
            seed=Value("int64"),
        ),
    )
    ds_id = "drawbench-sdxl"
    ds.push_to_hub(ds_id)


if __name__ == "__main__":
    main()
