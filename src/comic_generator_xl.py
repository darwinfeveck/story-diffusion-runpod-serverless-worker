import copy
import os
import pickle
import random
import requests
import sys
import numpy as np
import torch
import torch.nn.functional as F
from utils import is_torch2_available, cal_attn_mask_xl, setup_seed
from diffusers import DDIMScheduler, EulerDiscreteScheduler
from PIL import Image
from pipeline import StoryDiffusionXLPipeline

# Set global attention processor based on Torch version
if is_torch2_available():
    from utils import AttnProcessor2_0 as AttnProcessor
else:
    from utils import AttnProcessor

# Global variables for attention processing
_total_count = 0
_attn_count = 0
_cur_step = 0
_mask1024 = None
_mask4096 = None
_sa32 = 0.5
_sa64 = 0.5
_write = False
_height = 768
_width = 768

class SpatialAttnProcessor2_0(torch.nn.Module):
    def __init__(self, hidden_size=None, cross_attention_dim=None, id_length=4, device="cuda", dtype=torch.float16):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0+")
        
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        global _total_count, _attn_count, _cur_step, _mask1024, _mask4096
        global _sa32, _sa64, _write, _height, _width

        if _write:
            self.id_bank[_cur_step] = [hidden_states[:self.id_length], hidden_states[self.id_length:]]
        else:
            encoder_hidden_states = torch.cat((
                self.id_bank[_cur_step][0].to(self.device),
                hidden_states[:1],
                self.id_bank[_cur_step][1].to(self.device), 
                hidden_states[1:]
            ))

        if _cur_step < 5:
            hidden_states = self.__call2__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        else:
            random_number = random.random()
            rand_num = 0.3 if _cur_step < 20 else 0.1
            if random_number > rand_num:
                if not _write:
                    mask = _mask1024 if hidden_states.shape[1] == (_height//32)*(_width//32) else _mask4096
                    attention_mask = mask[mask.shape[0]//self.total_length*self.id_length:]
                else:
                    mask = _mask1024 if hidden_states.shape[1] == (_height//32)*(_width//32) else _mask4096
                    attention_mask = mask[:mask.shape[0]//self.total_length*self.id_length,
                                         :mask.shape[0]//self.total_length*self.id_length]
                hidden_states = self.__call1__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            else:
                hidden_states = self.__call2__(attn, hidden_states, None, attention_mask, temb)

        _attn_count += 1
        if _attn_count == _total_count:
            _attn_count = 0
            _cur_step += 1
            _mask1024, _mask4096 = cal_attn_mask_xl(
                self.total_length,
                self.id_length,
                _sa32,
                _sa64,
                _height,
                _width,
                device=self.device,
                dtype=self.dtype
            )
        return hidden_states

    # ... [Keep the existing __call1__ and __call2__ methods unchanged] ...

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

        # Load pipeline
        self.pipe = StoryDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        ).to(device)

        # Load PhotoMaker adapter with error handling
        photomaker_path = os.path.join(model_name, "photomaker", "photomaker-v2.bin")
        try:
            print(f"Loading PhotoMaker adapter from: {photomaker_path}")
            self.pipe.load_photomaker_adapter(
                photomaker_path,
                subfolder="",
                weight_name=os.path.basename(photomaker_path),
                trigger_word=self.trigger_word,
                strict=False  # Critical fix for loading mismatched checkpoints
            )
            print("PhotoMaker adapter loaded successfully")
        except Exception as e:
            print(f"Error loading PhotoMaker adapter: {e}")
            print("Continuing without PhotoMaker functionality")
            self.has_photomaker = False
        else:
            self.has_photomaker = True

        # Configure FreeU and scheduler
        self.pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        if scheduler_type == "euler":
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        else:
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.set_timesteps(50)

        # Configure attention processors
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            
            if cross_attention_dim is None and name.startswith("up_blocks"):
                attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
                _total_count += 1
            else:
                attn_procs[name] = AttnProcessor()

        print(f"Successfully loaded consistent self-attention with {_total_count} processors")
        unet.set_attn_processor(copy.deepcopy(attn_procs))

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
        global _sa32, _sa64, _height, _width, _write, _mask1024, _mask4096, _cur_step, _attn_count
        
        _sa32 = sa32
        _sa64 = sa64
        _height = height
        _width = width
        
        _write = False
        _mask1024, _mask4096 = cal_attn_mask_xl(
            self.total_length,
            self.id_length,
            _sa32,
            _sa64,
            _height,
            _width,
            device=self.device,
            dtype=self.torch_dtype
        )

        setup_seed(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        id_prompts = prompts[:self.id_length]
        real_prompts = prompts[self.id_length:]
        torch.cuda.empty_cache()

        _write = True
        _cur_step = 0
        _attn_count = 0
        input_id_images = [image_ref] if image_ref is not None else None

        try:
            id_images = self.pipe(
                id_prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                generator=generator,
                input_id_images=input_id_images
            ).images

            _write = False
            real_images = []
            for real_prompt in real_prompts:
                _cur_step = 0
                real_images.append(
                    self.pipe(
                        real_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        generator=generator,
                        input_id_images=input_id_images
                    ).images[0]
                )

            return id_images + real_images

        except Exception as e:
            print(f"Error during generation: {e}")
            raise