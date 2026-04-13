import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LLMQuantizer:
    def __init__(self, model_id, device="cuda"):
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def quantize_4bit(self):
        """Quantize model to 4-bit using NF4."""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        return model

    def quantize_8bit(self):
        """Quantize model to 8-bit."""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            load_in_8bit=True,
            device_map="auto"
        )
        return model

    def benchmark(self, model, prompt="Explain quantum computing in simple terms."):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Example usage
    quantizer = LLMQuantizer("meta-llama/Llama-2-7b-hf")
    # model = quantizer.quantize_4bit()
    # print(quantizer.benchmark(model))
