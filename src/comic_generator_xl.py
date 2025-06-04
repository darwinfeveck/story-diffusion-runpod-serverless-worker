import copy
import os
import torch
from PIL import Image
from diffusers import DDIMScheduler, EulerDiscreteScheduler
from pipeline import StoryDiffusionXLPipeline

# [Previous imports and SpatialAttnProcessor2_0 class remain the same...]

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
        global _total_count
        _total_count = 0
        
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

        # Try loading PhotoMaker adapter if available
        photomaker_path = os.path.join(model_name, "photomaker", "photomaker-v2.bin")
        if os.path.exists(photomaker_path):
            try:
                print(f"Attempting to load PhotoMaker from: {photomaker_path}")
                self.pipe.load_photomaker_adapter(
                    photomaker_path,
                    subfolder="",
                    weight_name=os.path.basename(photomaker_path),
                    trigger_word=self.trigger_word,
                    strict=False
                )
                self.has_photomaker = True
                print("PhotoMaker loaded successfully")
            except Exception as e:
                print(f"PhotoMaker loading failed, continuing without it: {str(e)}")
                self.has_photomaker = False

        # Configure pipeline
        self.pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        if scheduler_type == "euler":
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        else:
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # [Rest of the initialization remains the same...]

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
        image_ref: Image.Image = None
    ):
        try:
            # Initialize default return value
            result_images = []
            
            # Validate prompts for PhotoMaker if enabled
            if self.has_photomaker:
                for prompt in prompts:
                    if isinstance(prompt, str) and prompt.count(self.trigger_word) > 1:
                        raise ValueError(
                            f"PhotoMaker doesn't support multiple trigger words ('{self.trigger_word}') "
                            f"in a single prompt. Problematic prompt: {prompt}"
                        )

            # [Previous setup code for attention masks, etc...]

            # Generate ID images
            _write = True
            _cur_step = 0
            _attn_count = 0
            input_id_images = [image_ref] if image_ref is not None else None

            try:
                id_images = self.pipe(
                    prompts[:self.id_length],
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    negative_prompt=negative_prompt,
                    generator=torch.Generator(device=self.device).manual_seed(seed),
                    input_id_images=input_id_images
                ).images
                result_images.extend(id_images)
            except Exception as e:
                print(f"Error generating ID images: {e}")
                return None

            # Generate real images
            _write = False
            for real_prompt in prompts[self.id_length:]:
                try:
                    _cur_step = 0
                    image = self.pipe(
                        real_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        generator=torch.Generator(device=self.device).manual_seed(seed),
                        input_id_images=input_id_images
                    ).images[0]
                    result_images.append(image)
                except Exception as e:
                    print(f"Error generating image for prompt '{real_prompt}': {e}")
                    result_images.append(Image.new("RGB", (width, height), (0, 0, 0)))  # Return black image as placeholder

            return result_images

        except Exception as e:
            print(f"Error in ComicGeneratorXL: {e}")
            return None