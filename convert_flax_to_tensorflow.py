import numpy as np

import jax
import jax.numpy as jnp

from transformers import AutoTokenizer, TFGPTNeoForCausalLM

MODEL_PATH_TF = ""
MODEL_PATH_PT = ""


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_PT)
tokenizer.pad_token = tokenizer.eos_token

model_tf = TFGPTNeoForCausalLM.from_pretrained(MODEL_PAH_PT, from_pt=True)
model_tf.save_pretrained(MODEL_PATH_TF)


