import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

class QuantizationEngine:
    """
    Toolkit for optimizing Large Language Models using quantization techniques.
    """
    def __init__(self, model_id: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def load_4bit_model(self, use_double_quant: bool = True, bnb_4bit_quant_type: str = "nf4", 
                        bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=use_double_quant,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        return model

    def load_8bit_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            load_in_8bit=True,
            device_map="auto"
        )
        return model

    def generate_response(self, model: nn.Module, prompt: str, max_new_tokens: int = 100, 
                          temperature: float = 0.7, top_p: float = 0.9) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def benchmark_performance(self, model: nn.Module, prompt: str = "What is quantization?") -> Dict[str, Any]:
        import time
        start_time = time.time()
        _ = self.generate_response(model, prompt, max_new_tokens=10)
        latency = time.time() - start_time
        if self.device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        else:
            memory_allocated = 0.0
            memory_reserved = 0.0
        return {
            "latency": f"{latency:.4f}s",
            "memory_allocated_gb": f"{memory_allocated:.4f}GB",
            "memory_reserved_gb": f"{memory_reserved:.4f}GB"
        }

    def save_optimized_model(self, model: nn.Module, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    try:
        engine = QuantizationEngine("gpt2")
        print("Quantization Engine initialized for GPT-2.")
    except Exception as e:
        print(f"Initialization failed: {e}")
