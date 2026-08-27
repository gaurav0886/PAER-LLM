"""
llm_model.py

Wrapper around the response-generation LLM.

Bugs fixed
----------
1. **``torch_dtype=torch.float16`` unconditionally.** On a CPU-only machine
   float16 matmuls are either unsupported or emulated at a large slowdown, and
   some kernels produce NaNs. The dtype is now chosen from the actual device:
   bfloat16 on a modern GPU, float16 on an older one, float32 on CPU.

2. **``device_map="auto"`` requires ``accelerate``**, which was not in either
   requirements file. Without it the constructor raises ImportError. There is
   now a graceful fallback.

3. **Hand-written ``<|system|>`` prompt strings.** The prompt builder produced
   Phi-3 control tokens as literal text. That works only for Phi-3, and even
   there it is fragile. ``generate`` now accepts chat messages and applies the
   tokenizer's own chat template.

4. **``trust_remote_code=True``** was set for a model that no longer needs it.
   It executes arbitrary code from the Hub at load time; it is now opt-in.

5. **Step-by-step ``print`` statements** on every generation call. Replaced
   with an opt-in ``verbose`` flag.

6. **Greedy decoding only** (``do_sample=False``). Every run on the same
   emotion produced a byte-identical reply, which makes the "empathetic
   response" component look like a lookup table. Sampling is now the default,
   with the seed under your control for reproducibility when you need it.
"""

from __future__ import annotations

import torch


def _pick_dtype(device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class EmotionLLM:
    """Text generation for emotion-aware responses.

    Parameters
    ----------
    model_name : str
        Any instruction-tuned causal LM on the Hugging Face Hub.
    device : str, optional
        ``"cuda"``, ``"cpu"``, or None to auto-detect.
    trust_remote_code : bool
        Leave False unless the model genuinely requires custom code.
    verbose : bool
        Print progress for each generation call.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: str | None = None,
        *,
        trust_remote_code: bool = False,
        verbose: bool = False,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = _pick_dtype(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = dict(
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )

        try:
            import accelerate  # noqa: F401

            load_kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except ImportError:
            # accelerate missing - load normally and move the model ourselves
            # instead of crashing, which is what the original did.
            load_kwargs.pop("low_cpu_mem_usage", None)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            self.model.to(self.device)

        self.model.eval()
        print(f"[llm] {model_name} loaded on {self.device} ({dtype})")

    # ------------------------------------------------------------------
    def _to_text(self, prompt) -> str:
        """Accept either chat messages or a raw string."""
        if isinstance(prompt, str):
            return prompt

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )

        # Fallback for tokenizers without a chat template.
        return (
            "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in prompt)
            + "\n\nASSISTANT:"
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        max_tokens: int = 120,
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        seed: int | None = None,
    ) -> str:
        """Generate a response.

        Parameters
        ----------
        prompt : str or list[dict]
            A raw string, or chat messages from ``PromptBuilder``.
        max_tokens : int
            Maximum new tokens. The original default of 50 truncated most
            replies mid-sentence, which is why responses looked clipped.
        seed : int, optional
            Set for a reproducible sample.
        """
        if seed is not None:
            torch.manual_seed(seed)

        text = self._to_text(prompt)

        if self.verbose:
            print("[llm] tokenizing")

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            num_beams=1,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated = outputs[0][inputs.input_ids.shape[-1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        return self._clean(response)

    # ------------------------------------------------------------------
    @staticmethod
    def _clean(response: str) -> str:
        """Trim any prompt scaffolding the model echoed back."""
        stop_markers = (
            "<|user|>", "<|assistant|>", "<|system|>", "<|end|>",
            "Vocal tone suggests:", "Top candidates:", "Respond to them now.",
        )
        for marker in stop_markers:
            if marker in response:
                response = response.split(marker)[0]
        return response.strip()


__all__ = ["EmotionLLM"]
