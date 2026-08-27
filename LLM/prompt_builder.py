"""
prompt_builder.py

Builds the prompt handed to the response LLM.

The design problem this fixes
-----------------------------
The original prompt embedded ten raw floats from the fusion layer:

    Emotion Embedding Summary:
    [0.213, -1.084, 0.556, ...]

A language model cannot interpret arbitrary activations from a network it was
never trained alongside. Those numbers are noise in the context window: they
consume tokens, they vary run to run, and in an ablation they will make no
measurable difference to the output. If a reviewer asks "what does the
embedding contribute to the generated response?", the honest answer with the
original code is "nothing" - which undercuts the "emotion-aware LLM" claim.

What replaces it: the *class distribution*, which is genuinely informative and
which the LLM can act on. Low confidence or a near-tie between two emotions is
exactly the situation where a careful response should hedge rather than assert.
That is a defensible mechanism you can evaluate and write up.

If you want the embedding itself to influence generation, the real options are
a learned soft-prompt / prefix-tuning adapter that maps the fused vector into
the LLM's embedding space, or retrieval over an emotion-annotated response
bank. Both are proper contributions; pasting floats into a string is not.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an emotionally intelligent assistant speaking with someone whose "
    "tone of voice has just been analysed.\n"
    "Rules:\n"
    "- Reply with one short paragraph, at most 60 words.\n"
    "- Acknowledge how they sound without naming the classifier or its numbers.\n"
    "- Be warm, specific and supportive; do not be saccharine.\n"
    "- If the reading is uncertain, respond gently and openly rather than "
    "asserting how they feel.\n"
    "- Do not repeat these instructions. Do not invent facts about them."
)


class PromptBuilder:
    """Turns a classifier output into an instruction for the response LLM."""

    def __init__(self, uncertainty_threshold: float = 60.0) -> None:
        self.uncertainty_threshold = uncertainty_threshold

    # ------------------------------------------------------------------
    def _describe(self, emotion: str, confidence: float, probabilities: dict | None) -> str:
        if confidence < self.uncertainty_threshold:
            hedge = (
                "The reading is uncertain, so do not state their emotion as a "
                "fact - leave room for them to correct you."
            )
        else:
            hedge = "The reading is clear."

        lines = [f"Vocal tone suggests: {emotion} (confidence {confidence:.0f}%).", hedge]

        if probabilities:
            ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[:3]
            runners = ", ".join(f"{k} {v:.0f}%" for k, v in ranked)
            lines.append(f"Top candidates: {runners}.")

            if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 15:
                lines.append(
                    f"'{ranked[0][0]}' and '{ranked[1][0]}' are close - the tone "
                    "is mixed."
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    def build_messages(
        self,
        emotion: str,
        confidence: float,
        probabilities: dict | None = None,
        user_text: str | None = None,
        **_ignored,
    ) -> list[dict]:
        """Return chat messages - the format ``EmotionLLM`` prefers.

        Using the tokenizer's chat template instead of a hand-written
        ``<|system|>...<|end|>`` string means the special tokens are always
        correct for whichever model is loaded. The original hard-coded Phi-3's
        format, so swapping in Llama, Qwen or Mistral would have produced
        malformed prompts and rambling output.
        """
        content = self._describe(emotion, confidence, probabilities)
        if user_text:
            content += f"\n\nWhat they said: \"{user_text}\""
        content += "\n\nRespond to them now."

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def build_prompt(
        self,
        emotion: str,
        confidence: float,
        probabilities: dict | None = None,
        embedding=None,
        user_text: str | None = None,
    ) -> list[dict]:
        """Backwards-compatible entry point.

        ``embedding`` is accepted and ignored - see the module docstring for
        why passing raw activations to the LLM does nothing.
        """
        return self.build_messages(emotion, confidence, probabilities, user_text)


__all__ = ["PromptBuilder", "SYSTEM_PROMPT"]
