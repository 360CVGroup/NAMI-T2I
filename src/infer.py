import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MT5Tokenizer,MT5EncoderModel,BertModel, BertTokenizer
import diffusers
from scheduling_flow_matching import PyramidFlowMatchEulerDiscreteScheduler
from diffusers import (
    AutoencoderKL,
    FluxPipeline,
    FluxTransformer2DModel,
)

def _encode_prompt_with_mclip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask.bool()
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask.to(device),
    )

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds[0][:,[0]]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds

def _encode_prompt_with_mt5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            max_length=max_sequence_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt")
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_attention_mask = text_inputs.attention_mask
    prompt_attention_mask = prompt_attention_mask.to(device)

    prompt_embeds = text_encoder(
        text_input_ids.to(device),
        output_hidden_states=True,
    )
    prompt_embeds = prompt_embeds.hidden_states[-1] 

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    
    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
    prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)
    return prompt_embeds, prompt_attention_mask

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype
    device = device if device is not None else text_encoders[1].device
    pooled_prompt_embeds = _encode_prompt_with_mclip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds, prompt_attention_mask = _encode_prompt_with_mt5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    return prompt_embeds, pooled_prompt_embeds, prompt_attention_mask

class CustomFluxTransformer2DModel(FluxTransformer2DModel):
    def __init__(self):
        ######### 2B ##########
        super().__init__(
            patch_size = 1,
            in_channels = 64,
            num_layers = 8,
            num_single_layers = 14,
            attention_head_dim = 128,
            num_attention_heads = 16,
            joint_attention_dim = 4096,
            pooled_projection_dim = 1024,
            guidance_embeds = False,
            axes_dims_rope = [16, 56, 56],
            stage_singleblock_index = [1, 8, 14], ######layer ratio: 9:7:6
            )
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class BridgeFlow(nn.Module):
    def __init__(self):
        super(BridgeFlow, self).__init__()

        stage1_shift = nn.Parameter(torch.randn(1, 16, 64, 64))
        stage2_shift = nn.Parameter(torch.randn(1, 16, 128, 128))
        self.scale = torch.nn.Parameter(torch.randn(2))
        self.shift = nn.ParameterList([stage1_shift, stage2_shift])

    def forward(self, input_x, i_stage):
        x = input_x
        _, _, h, w = x.shape
        h *= 2
        w *= 2
        x = F.interpolate(x, size=(h, w), mode='nearest')
        ix = i_stage - 1
        out = self.scale[ix] * x + self.shift[ix]#.cuda().to(dtype=x.dtype)
        return out

def load_text_encoders(weights_dir, device, dtype):
    tokenizer_one = BertTokenizer.from_pretrained(f"{weights_dir}/mclip/tokenizer")
    text_encoder_one = BertModel.from_pretrained(
        f"{weights_dir}/mclip/clip_text_encoder",
        add_pooling_layer=False,
        revision=None
    )

    tokenizer_two = MT5Tokenizer.from_pretrained(f"{weights_dir}/mt5")
    text_encoder_two = MT5EncoderModel.from_pretrained(f"{weights_dir}/mt5")

    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    
    text_encoder_one = text_encoder_one.to(device, dtype=dtype)
    text_encoder_two = text_encoder_two.to(device, dtype=dtype)
    
    return tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two


def load_models(weights_dir, device, dtype):
    vae = AutoencoderKL.from_pretrained(f"{weights_dir}/vae")
    vae.requires_grad_(False)
    vae = vae.to(device, dtype=dtype)
    
    transformer = CustomFluxTransformer2DModel()
    transformer.load_state_dict(torch.load(f"{weights_dir}/transformer/pytorch_model_fsdp.bin"))
    transformer = transformer.to(device, dtype=dtype)
    
    bridge_flow = BridgeFlow()
    bridge_flow.load_state_dict(torch.load(f"{weights_dir}/bridge_flow.bin"))
    bridge_flow = bridge_flow.to(device, dtype=dtype)
    bridge_flow.eval()

    return vae, transformer, bridge_flow


def build_scheduler():
    timestep_shift = 1.0
    stages = [1, 2, 4]
    stage_range = [0, 1/3, 2/3, 1]
    scheduler_gamma = 1/3

    scheduler = PyramidFlowMatchEulerDiscreteScheduler(
        shift=timestep_shift,
        stages=len(stages),
        stage_range=stage_range,
        gamma=scheduler_gamma,
    )

    return scheduler, stages


def build_pipeline(
    vae,
    transformer,
    text_encoder,
    tokenizer,
    text_encoder_2,
    tokenizer_2,
    bridge_flow,
    scheduler,
    stages,
    device,
    dtype
):
    pipe = FluxPipeline(
        scheduler=scheduler,
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        stages=stages,
        bridge_flow=bridge_flow,
    ).to(device, dtype=dtype)

    return pipe

def main(weights_dir):
    print ('build pipeline start')
    weights_dir = weights_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16
    
    tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two = load_text_encoders(weights_dir, device, weight_dtype)
    print ('build text encoder done')
    vae, transformer, bridge_flow = load_models(weights_dir, device, weight_dtype)
    print ('build model done')
    scheduler, stages = build_scheduler()

    pipe = build_pipeline(
        vae,
        transformer,
        text_encoder_one,
        tokenizer_one,
        text_encoder_two,
        tokenizer_two,
        bridge_flow,
        scheduler,
        stages,
        device,
        weight_dtype
    )
    print ('build pipeline done')
    common_pos = ".extremely detailed,best quality,high quality,perfect composition"
    common_neg = "(face asymmetry, eyes asymmetry, deformed eyes, open mouth)"

    caption = "a photo of a motorcycle"

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = encode_prompt(
            [text_encoder_one, text_encoder_two],
            [tokenizer_one, tokenizer_two],
            [caption + common_pos, common_neg],
            256
        )

        prompt_embeds, prompt_embeds_ng = prompt_embeds.chunk(2)
        pooled_prompt_embeds, pooled_prompt_embeds_ng = pooled_prompt_embeds.chunk(2)
        prompt_attention_mask, prompt_attention_mask_ng = prompt_attention_mask.chunk(2)
        print ('pipeline start')
        image = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=prompt_embeds_ng,
            negative_pooled_prompt_embeds=pooled_prompt_embeds_ng,
            height=1024,
            width=1024,
            guidance_scale=3,
            num_inference_steps=[10, 10, 10],
            max_sequence_length=256,
            if_use_pool=True,
        ).images[0]
        print ('generate image done')
        image.save("temp.png")

 
main("weights")