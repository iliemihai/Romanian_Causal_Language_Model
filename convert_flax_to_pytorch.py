import torch
import numpy as np

import jax
import jax.numpy as jnp

from transformers import AutoTokenizer, FlaxGPTNeoForCausalLM, GPTNeoForCausalLM

MODEL_PATH_FX = ""
MODEL_PATH_PT = ""

# if model weights are bfloat16 convert to float32
def to_f32(t):
    return jax.tree_map(lambda x: x.astype(jnp.float32) if x.dtype==jnp.bfloat16 else x, t)

#model_fx.params = to_f32(model_fx.params)
#model_fx.save_pretrained(MODEL_PATH_FX)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_FX)
tokenizer.pad_token = tokenizer.eos_token

model_fx = FlaxGPTNeoForCausalLM.from_pretrained(MODEL_PATH_FX)

model_pt = GPTNeoForCausalLM.from_pretrained(MODEL_PAH_FX, from_flax=True)
model_pt.save_pretrained(MODEL_PATH_PT)


