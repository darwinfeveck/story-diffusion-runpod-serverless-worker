import torch
import numpy as np
from typing import List, Optional, Union, Tuple
from PIL import Image
from diffusers import EulerDiscreteScheduler
from diffusers.utils import load_image
from transformers import AutoTokenizer, CLIPImageProcessor
import logging
import os

# Import PhotoMaker V2 pipeline (assuming it's in the same directory or installed)
from pipeline_t2i_adapter import PhotoMakerStableDiffusionXLAdapterPipeline
from style_template import styles

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComicGeneratorXL:
    def __init__(
        self,
        model_name: str = "SG161222/RealVisXL_V4.0",
        id_length: int = 4,
        total_length: int = 5,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        scheduler_type: str = "euler",
        trigger_word: str = "img",
        photomaker_version: str = "v2",
        adapter_model: str = "TencentARC/t2i-adapter-sketch-sdxl-1.0",
        photomaker_ckpt: str = "TencentARC/PhotoMaker/photomaker-v2.bin"
    ):
        """
        Initialize the ComicGeneratorXL class with PhotoMaker V2 pipeline.

        Args:
            model_name (str): Base model name (e.g., RealVisXL_V4.0).
            id_length (int): Number of ID images to generate.
            total_length (int): Total number of images in the comic.
            device (str): Device to run the model on (cuda, cpu, mps).
            torch_dtype (torch.dtype): Data type for model weights.
            scheduler_type (str): Scheduler type (e.g., euler).
            trigger_word (str): Trigger word for PhotoMaker (default: 'img').
            photomaker_version (str): PhotoMaker version (v1 or v2).
            adapter_model (str): T2I-Adapter model for sketch control.
            photomaker_ckpt (str): Path or repo ID for PhotoMaker checkpoint.
        """
        self.model_name = model_name
        self.id_length = id_length
        self.total_length = total_length
        self.device = device
        self.torch_dtype = torch_dtype
        self.trigger_word = trigger_word
        self.has_photomaker = False
        self.photomaker_version = photomaker_version

        # Determine device and dtype
        if self.device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
            self.torch_dtype = torch.float16
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32

        logger.info(f"Initializing ComicGeneratorXL on device: {self.device}, dtype: {self.torch_dtype}")

        # Load tokenizer explicitly
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.add_tokens([self.trigger_word], special_tokens=True)
        logger.info(f"Added trigger word '{self.trigger_word}' to tokenizer. Token ID: {self.tokenizer.convert_tokens_to_ids(self.trigger_word)}")

        # Initialize image processor
        self.id_image_processor = CLIPImageProcessor()

        # Load T2I-Adapter
        from diffusers import T2IAdapter
        self.adapter = T2IAdapter.from_pretrained(
            adapter_model,
            torch_dtype=self.torch_dtype,
            variant="fp16"
        ).to(self.device)
        logger.info(f"Loaded T2I-Adapter from {adapter_model}")

        # Load pipeline with the initialized tokenizer
        self.pipe = PhotoMakerStableDiffusionXLAdapterPipeline.from_pretrained(
            model_name,
            tokenizer=self.tokenizer,  # Pass the initialized tokenizer
            adapter=self.adapter,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
            variant="fp16"
        ).to(self.device)
        logger.info(f"Loaded pipeline from {model_name}")

        # Load PhotoMaker adapter
        self.pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_ckpt) if os.path.exists(photomaker_ckpt) else "TencentARC/PhotoMaker",
            subfolder="",
            weight_name=os.path.basename(photomaker_ckpt) if os.path.exists(photomaker_ckpt) else "photomaker-v2.bin",
            trigger_word=self.trigger_word,
            pm_version=self.photomaker_version
        )
        self.has_photomaker = True
        logger.info(f"Loaded PhotoMaker {self.photomaker_version} adapter with trigger word: {self.trigger_word}")

        # Move ID encoder to device
        self.pipe.id_encoder.to(self.device)
        logger.info("Moved ID encoder to device")

        # Set scheduler
        if scheduler_type == "euler":
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        logger.info(f"Set scheduler to {scheduler_type}")

        # Fuse LoRA weights
        self.pipe.fuse_lora()
        logger.info("Fused LoRA weights")

    def apply_style(self, style_name: str, prompt: str, negative_prompt: str = "") -> Tuple[str, str]:
        """
        Apply a style template to the prompt and negative prompt.

        Args:
            style_name (str): Name of the style (e.g., 'Comic book').
            prompt (str): Original prompt.
            negative_prompt (str): Original negative prompt.

        Returns:
            Tuple[str, str]: Modified prompt and negative prompt.
        """
        if style_name not in styles:
            logger.warning(f"Style '{style_name}' not found. Using default 'Photographic' style.")
            style_name = "Photographic (Default)"
        
        style_prompt, style_neg_prompt = styles[style_name]
        return (
            style_prompt.replace("{prompt}", prompt),
            style_neg_prompt + (" " + negative_prompt if negative_prompt else "")
        )

    def preprocess_prompts(self, prompts: Union[str, List[str]]) -> List[str]:
        """
        Preprocess prompts to ensure they are valid and contain the trigger word.

        Args:
            prompts (Union[str, List[str]]): Input prompt(s).

        Returns:
            List[str]: List of valid prompts.

        Raises:
            ValueError: If any prompt is invalid (not a string, empty, or lacks the trigger word).
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if not prompts:
            raise ValueError("No prompts provided. At least one prompt with the trigger word is required.")

        valid_prompts = []
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                raise ValueError(f"Prompt at index {i} is not a string: {prompt}")
            prompt = prompt.strip()
            if not prompt:
                raise ValueError(f"Empty prompt at index {i}")
            if self.trigger_word not in prompt:
                raise ValueError(f"Trigger word '{self.trigger_word}' not in prompt at index {i}: {prompt}")
            valid_prompts.append(prompt)
        
        return valid_prompts

    def __call__(
        self,
        prompts: Union[str, List[str]],
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        height: int = 1024,
        width: int = 1024,
        seed: Optional[int] = None,
        style_name: str = "Comic book",
        input_id_images: Optional[List[Union[str, Image.Image]]] = None,
        use_doodle: bool = False,
        sketch_image: Optional[Image.Image] = None,
        adapter_conditioning_scale: float = 0.7,
        adapter_conditioning_factor: float = 0.8,
        style_strength_ratio: float = 20
    ) -> List[Image.Image]:
        """
        Generate a comic sequence using PhotoMaker V2 pipeline.

        Args:
            prompts (Union[str, List[str]]): Prompt(s) for image generation.
            negative_prompt (str): Negative prompt to avoid certain features.
            num_inference_steps (int): Number of denoising steps.
            guidance_scale (float): Guidance scale for classifier-free guidance.
            height (int): Output image height.
            width (int): Output image width.
            seed (Optional[int]): Random seed for reproducibility.
            style_name (str): Style template name (e.g., 'Comic book').
            input_id_images (Optional[List[Union[str, Image.Image]]]): Input ID images for personalization.
            use_doodle (bool): Enable sketch-based control using T2I-Adapter.
            sketch_image (Optional[Image]): Sketch image for doodle control.
            adapter_conditioning_scale (float): Conditioning scale for T2I-Adapter.
            adapter_conditioning_factor (float): Fraction of timesteps for adapter conditioning.
            style_strength_ratio (float): Percentage of steps for style fusion.

        Returns:
            List[Image.Image]: List of PIL images for the comic sequence.
        """
        # Set random seed
        if seed is None:
            seed = np.random.randint(0, np.iinfo(np.int32).max)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        logger.info(f"Using seed: {seed}")

        # Preprocess prompts
        prompts = self.preprocess_prompts(prompts)
        logger.info(f"Valid prompts: {prompts}")

        # Validate input ID images
        if input_id_images is None or not input_id_images:
            logger.error("No input ID images provided.")
            raise ValueError("At least one input ID image is required.")

        input_id_images_processed = []
        for img in input_id_images:
            if isinstance(img, str):
                img = load_image(img)
            if not isinstance(img, Image.Image):
                logger.warning(f"Invalid ID image type: {type(img)}. Skipping.")
                continue
            input_id_images_processed.append(img)
        
        if not input_id_images_processed:
            logger.error("No valid ID images provided.")
            raise ValueError("No valid ID images provided.")

        logger.info(f"Using {len(input_id_images_processed)} ID images")

        # Preprocess sketch image for doodle
        if use_doodle and sketch_image is not None:
            if isinstance(sketch_image, dict):
                sketch_image = sketch_image.get("composite", sketch_image)
            if isinstance(sketch_image, Image.Image):
                r, g, b, a = sketch_image.split()
                sketch_image = a.convert("RGB")
                sketch_image = torch.from_numpy(np.array(sketch_image)).float() / 255.0
                sketch_image = sketch_image > 0.5  # Inversion
                sketch_image = Image.fromarray(sketch_image.numpy().astype(np.uint8) * 255)
            else:
                logger.warning("Invalid sketch image. Disabling doodle.")
                use_doodle = False
        else:
            use_doodle = False
            sketch_image = None
            adapter_conditioning_scale = 0.0
            adapter_conditioning_factor = 0.0

        logger.info(f"Doodle enabled: {use_doodle}, Adapter scale: {adapter_conditioning_scale}, Factor: {adapter_conditioning_factor}")

        # Calculate start merge step
        start_merge_step = int((style_strength_ratio / 100) * num_inference_steps)
        if start_merge_step > 30:
            start_merge_step = 30
        logger.info(f"Start merge step: {start_merge_step}")

        # Initialize result images
        result_images = []

        # Generate ID images (first id_length images)
        id_prompts = prompts[:min(self.id_length, len(prompts))]
        for i, id_prompt in enumerate(id_prompts):
            logger.info(f"Processing ID prompt {i+1}/{len(id_prompts)}: '{id_prompt}'")
            
            try:
                # Apply style template
                styled_prompt, styled_neg_prompt = self.apply_style(style_name, id_prompt, negative_prompt)
                logger.debug(f"Styled prompt: {styled_prompt}")
                logger.debug(f"Styled negative prompt: {styled_neg_prompt}")

                # Generate image
                images = self.pipe(
                    prompt=styled_prompt,
                    negative_prompt=styled_neg_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                    input_id_images=input_id_images_processed,
                    start_merge_step=start_merge_step,
                    num_images_per_prompt=1,
                    image=sketch_image if use_doodle else None,
                    adapter_conditioning_scale=adapter_conditioning_scale,
                    adapter_conditioning_factor=adapter_conditioning_factor
                ).images

                result_images.extend(images)
            except Exception as e:
                logger.error(f"Failed to generate ID image for prompt '{id_prompt}': {str(e)}")
                result_images.append(Image.new("RGB", (width, height), (0, 0, 0)))

        # Generate remaining images (if total_length > id_length)
        remaining_prompts = prompts[self.id_length:self.total_length] if len(prompts) > self.id_length else []
        for i, prompt in enumerate(remaining_prompts):
            logger.info(f"Processing remaining prompt {i+1}/{len(remaining_prompts)}: '{prompt}'")
            
            try:
                # Apply style template
                styled_prompt, styled_neg_prompt = self.apply_style(style_name, prompt, negative_prompt)
                logger.debug(f"Styled prompt: {styled_prompt}")
                logger.debug(f"Styled negative prompt: {styled_neg_prompt}")

                # Generate image
                images = self.pipe(
                    prompt=styled_prompt,
                    negative_prompt=styled_neg_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                    input_id_images=input_id_images_processed,
                    start_merge_step=start_merge_step,
                    num_images_per_prompt=1,
                    image=sketch_image if use_doodle else None,
                    adapter_conditioning_scale=adapter_conditioning_scale,
                    adapter_conditioning_factor=adapter_conditioning_factor
                ).images

                result_images.extend(images)
            except Exception as e:
                logger.error(f"Failed to generate image for prompt '{prompt}': {str(e)}")
                result_images.append(Image.new("RGB", (width, height), (0, 0, 0)))

        # Pad with blank images if necessary
        while len(result_images) < self.total_length:
            logger.warning(f"Padding with blank image to reach total_length={self.total_length}")
            result_images.append(Image.new("RGB", (width, height), (0, 0, 0)))

        logger.info(f"Generated {len(result_images)} images")
        return result_images