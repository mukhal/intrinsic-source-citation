
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

class LlamaForCausalLMFSDP(LlamaForCausalLM):

    def fsdp_wrap_fn(self, module):
        return isinstance(module, LlamaDecoderLayer)

    def activation_checkpointing_fn(self, module):
        return isinstance(module, LlamaDecoderLayer)