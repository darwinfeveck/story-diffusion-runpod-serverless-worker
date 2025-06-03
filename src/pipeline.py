from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from collections import OrderedDict
import os
import PIL
import numpy as np

import torch
from torchvision import transforms as T

from safetensors import safe_open
from huggingface_hub.utils import validate_hf_hub_args
from transformers import CLIPImageProcessor, CLIPTokenizer
from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.utils import (
    _get_model_file,
    is_transformers_available,
    logging,
)

from photomaker_id_encoder import PhotoMakerIDEncoder
from utils import remove_word

PipelineImageInput = Union[
    PIL.Image.Image,
    torch.FloatTensor,
    List[PIL.Image.Image],
    List[torch.FloatTensor],
]


class StoryDiffusionXLPipeline(StableDiffusionXLPipeline):
    @validate_hf_hub_args
    def load_photomaker_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        weight_name: str,
        subfolder: str = '',
        trigger_word: str = 'img',
        **kwargs,
    ):
        """
        Load PhotoMaker v2 adapter components
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            
            print(f"Loading PhotoMaker v2 weights from: {model_file}")
            
            if weight_name.endswith(".safetensors"):
                state_dict = {"id_encoder": {}, "lora_weights": {}}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        # Keep original v2 key names
                        if key.startswith("id_encoder.") or key.startswith("qformer_perceiver."):
                            state_dict["id_encoder"][key] = f.get_tensor(key)
                        elif key.startswith("lora_weights."):
                            state_dict["lora_weights"][key.replace("lora_weights.", "")] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        # Verify required components
        if not all(k in state_dict for k in ["id_encoder", "lora_weights"]):
            raise ValueError("State dict must contain both 'id_encoder' and 'lora_weights'")

        self.trigger_word = trigger_word
        
        # Load ID Encoder (v2)
        print("Loading PhotoMaker v2 ID Encoder...")
        id_encoder = PhotoMakerIDEncoder()
        
        # Load with strict=False to handle potential architecture differences
        missing_keys, unexpected_keys = id_encoder.load_state_dict(
            state_dict["id_encoder"], 
            strict=False
        )
        
        if missing_keys:
            print(f"Warning: Missing keys in ID Encoder: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in ID Encoder: {unexpected_keys}")
            
        id_encoder = id_encoder.to(self.device)
        self.id_encoder = id_encoder

        # Initialize image processor
        self.id_image_processor = CLIPImageProcessor()

        # Load LoRA weights
        print("Loading PhotoMaker LoRA weights...")
        self.load_lora_weights(state_dict["lora_weights"], adapter_name="photomaker")

        # Add trigger word tokens
        if self.tokenizer is not None:
            self.tokenizer.add_tokens([self.trigger_word], special_tokens=True)
        self.tokenizer_2.add_tokens([self.trigger_word], special_tokens=True)

    def encode_prompt_with_trigger_word(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        num_id_images: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        class_tokens_mask: Optional[torch.LongTensor] = None,
    ):
        device = device or self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Get trigger word token ID
        image_token_id = self.tokenizer_2.convert_tokens_to_ids(self.trigger_word)

        # Prepare tokenizers and encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                input_ids = tokenizer.encode(prompt)
                clean_index = 0
                clean_input_ids = []
                class_token_index = []
                
                # Find trigger word and class token positions
                for i, token_id in enumerate(input_ids):
                    if token_id == image_token_id:
                        class_token_index.append(clean_index - 1)
                    else:
                        clean_input_ids.append(token_id)
                        clean_index += 1

                if len(class_token_index) != 1:
                    raise ValueError(
                        f"PhotoMaker requires exactly one trigger word. Found {len(class_token_index)} in: {prompt}"
                    )
                
                class_token_index = class_token_index[0]
                class_token = clean_input_ids[class_token_index]
                
                # Expand class token
                clean_input_ids = (
                    clean_input_ids[:class_token_index] + 
                    [class_token] * num_id_images + 
                    clean_input_ids[class_token_index+1:]
                )

                # Handle truncation/padding
                max_len = tokenizer.model_max_length
                if len(clean_input_ids) > max_len:
                    clean_input_ids = clean_input_ids[:max_len]
                else:
                    clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (
                        max_len - len(clean_input_ids))
                
                # Create mask for class tokens
                class_tokens_mask = [
                    True if class_token_index <= i < class_token_index+num_id_images 
                    else False for i in range(len(clean_input_ids))
                ]

                clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long).unsqueeze(0)
                class_tokens_mask = torch.tensor(class_tokens_mask, dtype=torch.bool).unsqueeze(0)

                # Get embeddings
                prompt_embeds = text_encoder(
                    clean_input_ids.to(device),
                    output_hidden_states=True,
                )

                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        if class_tokens_mask is not None:
            class_tokens_mask = class_tokens_mask.to(device=device)

        return prompt_embeds, pooled_prompt_embeds, class_tokens_mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # PhotoMaker-specific parameters
        input_id_images: PipelineImageInput = None,
        start_merge_step: int = 0,
        class_tokens_mask: Optional[torch.LongTensor] = None,
        prompt_embeds_text_only: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds_text_only: Optional[torch.FloatTensor] = None,
    ):
        # Handle case without ID images
        if input_id_images is None:
            prompt = remove_word(prompt, self.trigger_word)
            prompt_2 = remove_word(prompt_2, self.trigger_word) if prompt_2 else None
            negative_prompt = remove_word(negative_prompt, self.trigger_word) if negative_prompt else None
            negative_prompt_2 = remove_word(negative_prompt_2, self.trigger_word) if negative_prompt_2 else None
            
            return super().__call__(
                prompt=prompt,
                prompt_2=prompt_2,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                denoising_end=denoising_end,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                cross_attention_kwargs={"scale": 0.0},  # Disable PhotoMaker adapter
                guidance_rescale=guidance_rescale,
                original_size=original_size,
                crops_coords_top_left=crops_coords_top_left,
                target_size=target_size,
                callback=callback,
                callback_steps=callback_steps,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs
            )

        # Prepare ID images
        if not isinstance(input_id_images, list):
            input_id_images = [input_id_images]

        # Prepare timesteps
        device = self._execution_device
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Encode prompt with trigger word
        prompt_embeds, pooled_prompt_embeds, class_tokens_mask = self.encode_prompt_with_trigger_word(
            prompt=prompt,
            prompt_2=prompt_2,
            num_id_images=len(input_id_images),
            device=device,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            class_tokens_mask=class_tokens_mask,
        )

        # Prepare ID images
        dtype = next(self.id_encoder.parameters()).dtype
        if not isinstance(input_id_images[0], torch.Tensor):
            id_pixel_values = self.id_image_processor(input_id_images, return_tensors="pt").pixel_values
        id_pixel_values = id_pixel_values.unsqueeze(0).to(device=device, dtype=dtype)

        # Get ID embeddings and fuse with prompt
        prompt_embeds = self.id_encoder(
            id_pixel_values=id_pixel_values,
            prompt_embeds=prompt_embeds,
            class_tokens_mask=class_tokens_mask,
        )

        # Duplicate for multiple images per prompt
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

        # Prepare negative prompts
        negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
            negative_prompt,
            negative_prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=True,
        )

        # Prepare latents
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            bs_embed * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare added time ids
        text_encoder_projection_dim = (
            self.text_encoder_2.config.projection_dim
            if self.text_encoder_2 is not None
            else pooled_prompt_embeds.shape[-1]
        )
        add_time_ids = self._get_add_time_ids(
            original_size or (height, width),
            crops_coords_top_left,
            target_size or (height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
        add_time_ids = add_time_ids.to(device).repeat(bs_embed * num_images_per_prompt, 1)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Prepare model input
                latent_model_input = (
                    torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Select prompt embeddings based on merge step
                if i <= start_merge_step:
                    current_prompt_embeds = torch.cat(
                        [negative_prompt_embeds, prompt_embeds_text_only], dim=0
                    )
                    add_text_embeds = torch.cat(
                        [negative_pooled_prompt_embeds, pooled_prompt_embeds_text_only], dim=0
                    )
                else:
                    current_prompt_embeds = torch.cat(
                        [negative_prompt_embeds, prompt_embeds], dim=0
                    )
                    add_text_embeds = torch.cat(
                        [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
                    )

                # Predict noise
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=current_prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # Perform guidance
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if guidance_scale > 1.0 and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # Compute previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)[0]

                # Call callback
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # Decode latents
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        if output_type != "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = latents

        # Offload models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)