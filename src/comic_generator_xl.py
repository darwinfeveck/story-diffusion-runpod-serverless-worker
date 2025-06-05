import os
import torch
from PIL import Image
from diffusers import DDIMScheduler, EulerDiscreteScheduler
from pipeline import StoryDiffusionXLPipeline  # Your custom pipeline
from transformers import AutoTokenizer

class ComicGeneratorXL:
    def __init__(
        self,
        model_name: str,
        id_length: int = 4,
        total_length: int = 5,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        scheduler_type: str = "euler",
        trigger_word: str = "img",
    ):
        self.model_name = model_name
        self.id_length = id_length
        self.total_length = total_length
        self.device = device
        self.torch_dtype = torch_dtype
        self.trigger_word = trigger_word
        self.has_photomaker = False

        # Load pipeline
        self.pipe = StoryDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        ).to(device)

        # Remove the fallback tokenizer loading - let the pipeline use its default tokenizers
        # The original tokenizers from the model should work correctly
        
        # Load PhotoMaker V2 adapter
        photomaker_path = os.path.join(model_name, "photomaker", "photomaker-v2.bin")
        if os.path.exists(photomaker_path):
            try:
                print(f"Attempting to load PhotoMaker from: {photomaker_path}")
                self.pipe.load_photomaker_adapter(
                    pretrained_model_name_or_path_or_dict=os.path.dirname(photomaker_path),
                    subfolder="",
                    weight_name=os.path.basename(photomaker_path),
                    trigger_word=self.trigger_word,
                    strict=False,
                )
                self.has_photomaker = True
                print("PhotoMaker V2 loaded successfully.")
            except Exception as e:
                print(f"PhotoMaker loading failed: {e}")

        self.pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config) if scheduler_type == "euler" else DDIMScheduler.from_config(self.pipe.scheduler.config)

    def __call__(
        self,
        prompts: list,
        negative_prompt: str,
        width: int = 768,
        height: int = 768,
        sa32: float = 0.5,
        sa64: float = 0.5,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 50,
        seed: int = 2047,
        image_ref: Image.Image = None,
        style_strength_ratio: float = 0.2
    ):
        result_images = []

        # Ensure prompts are strings and properly formatted
        if not isinstance(prompts, list):
            prompts = [prompts]
        
        # Convert all prompts to strings and validate
        clean_prompts = []
        for prompt in prompts:
            if prompt is None:
                clean_prompt = ""
            elif isinstance(prompt, str):
                clean_prompt = prompt.strip()
            else:
                clean_prompt = str(prompt).strip()
            clean_prompts.append(clean_prompt)
        
        prompts = clean_prompts

        # Ensure negative prompt is a string
        if negative_prompt is None:
            negative_prompt = ""
        elif not isinstance(negative_prompt, str):
            negative_prompt = str(negative_prompt)
        negative_prompt = negative_prompt.strip()

        if self.has_photomaker:
            for prompt in prompts:
                if isinstance(prompt, str) and prompt.count(self.trigger_word) > 1:
                    print(f"[ERROR] Prompt contains multiple trigger words: '{prompt}'")
                    return []

        input_id_images = [image_ref] if image_ref is not None else None
        generator = torch.Generator(device=self.device).manual_seed(seed)
        start_merge_step = int(style_strength_ratio * num_inference_steps)
        start_merge_step = min(start_merge_step, 30)

        try:
            id_prompts = prompts[:self.id_length]
            for i, id_prompt in enumerate(id_prompts):
                print(f"Processing ID prompt {i+1}/{len(id_prompts)}: '{id_prompt}'")
                print(f"Prompt type: {type(id_prompt)}, content: {id_prompt}")
                print(f"Negative prompt type: {type(negative_prompt)}, content: {negative_prompt}")
                
                # Ensure we have valid string inputs
                if not isinstance(id_prompt, str) or not id_prompt.strip():
                    print(f"[WARNING] Empty or invalid prompt at index {i}, skipping...")
                    continue
                    
                id_images = self.pipe(
                    prompt=id_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                    input_id_images=input_id_images,
                    start_merge_step=0,
                    num_images_per_prompt=1,
                ).images
                result_images.extend(id_images)
        except Exception as e:
            print(f"[ERROR] Failed to generate ID images: {e}")
            print(f"[DEBUG] ID prompts: {prompts[:self.id_length]}")
            print(f"[DEBUG] Negative prompt: {negative_prompt}")
            return []

        # Generate remaining images
        for i, real_prompt in enumerate(prompts[self.id_length:], self.id_length):
            try:
                print(f"Processing prompt {i+1}/{len(prompts)}: '{real_prompt}'")
                
                # Ensure we have valid string inputs
                if not isinstance(real_prompt, str) or not real_prompt.strip():
                    print(f"[WARNING] Empty or invalid prompt at index {i}, creating blank image...")
                    result_images.append(Image.new("RGB", (width, height), (0, 0, 0)))
                    continue
                    
                image = self.pipe(
                    prompt=real_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                    input_id_images=input_id_images,
                    start_merge_step=start_merge_step,
                    num_images_per_prompt=1,
                ).images[0]
                result_images.append(image)
            except Exception as e:
                print(f"[ERROR] Failed to generate image for prompt '{real_prompt}': {e}")
                result_images.append(Image.new("RGB", (width, height), (0, 0, 0)))

        return result_images