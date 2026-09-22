"""
DocuMind W: Unified Grounded Reasoning Service (Qwen2.5-1.5B)
Hardened for enterprise document intelligence and small parameter models (SLMs).

Enforces:
1. Zero-Preamble System Prompt: Direct factual verdict or structured answer in Sentence 1.
   No procedural preambles ('Step 1: Extract...'), conversational pleasantries, or emojis.
2. Anti-Premise Echoing & Strict Negative Restraint:
   If data is absent, output 'Not Mentioned in Provided Context' in Sentence 1 and terminate.
   Forbid repeating ungrounded user premises.
3. Stop-Token & Delimiter Hardening:
   Native ChatML formatting, StopSequenceCriteria (['<|im_end|>', '<|endoftext|>', '\\nHuman:', '\\nUser:', '\\nAssistant:']),
   low temperature (0.05), and robust token budgeting (>= 896 tokens).
"""

import re
import logging
from typing import Any, Dict, List, Optional
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

logger = logging.getLogger("documind.services.rag")


class StopSequenceCriteria(StoppingCriteria):
    """
    Stops generation when any configured stop string appears in the generated sequence.
    Provides a multi-layered defense against role-hallucination and infinite generation.
    """
    def __init__(self, tokenizer, stop_strings: List[str], prompt_length: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check generated tokens only
        generated_ids = input_ids[0, self.prompt_length:]
        if len(generated_ids) == 0:
            return False
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        for s in self.stop_strings:
            if s in decoded:
                return True
        return False


class GroundedReasoningEngine:
    """
    Unified grounded reasoning engine supporting:
    - Language interpretation & contextual synthesis (Qwen2.5-1.5B)
    - Native Qwen ChatML template formatting
    - Strict claim-level negative restraint ('Not Mentioned in Provided Context')
    - Anti-premise echoing (verdict first, never adopt unsupported premises)
    - Multi-layered EOS and sequence termination criteria
    - Task-adaptive token budgeting (at least 896 tokens) and near-deterministic temperature (0.05)
    - Deterministic context injection from verified structured data
    """

    SYSTEM_PROMPT = """You are the grounded document intelligence engine of DocuMind W. You analyze retrieved enterprise documents with strict mathematical and factual adherence.

### CORE OPERATING RULES:
1. ZERO PREAMBLE & DIRECT ANSWER:
   - Directly state the factual verdict or structured answer in Sentence 1.
   - Do NOT state what steps you will take.
   - Do NOT output procedural preambles ('Step 1: Extract...', 'Let us begin by...').
   - Do NOT include conversational pleasantries, explanations of your logic, apologies, or emojis.

2. ABSOLUTE RULE ON PLACEHOLDER NUMBERS:
   - ABSOLUTE RULE: NEVER generate hypothetical, estimated, or placeholder numbers ($1,000,000, $500,000, etc.). If exact milestone release figures or requested fields are not verbatim in the context, output ONLY: 'Not Mentioned in Provided Context'.

3. PREMISE PROTECTION & VERIFICATION FIRST:
   - When asked about a causal relationship, approval, or event (e.g., "Why did X authorize Y?"), FIRST verify whether the premise is explicitly supported by the evidence.
   - If the document does NOT support the premise, state the verdict immediately in Sentence 1:
     "The provided context does not state that [X authorized Y]."
   - NEVER repeat an unverified user premise as an established fact.

4. STRICT NEGATIVE RESTRAINT:
   - If requested data or a specific field is absent from the provided context, output:
     "Not Mentioned in Provided Context"
     in Sentence 1 and terminate generation.
   - NEVER assume, speculate, or fabricate placeholder values (such as round increments of $15k, $20k, $25k).

5. SYNTHESIS OVER RE-COMPUTATION:
   - When verified structured extractions or deterministic calculation results are provided, cite and explain them. Do not re-perform arithmetic independently. Provide clean, concise markdown."""

    STOP_STRINGS = [
        "<|im_end|>",
        "<|endoftext|>",
        "\nHuman:",
        "\nUser:",
        "\nAssistant:",
        "Human:",
        "User:",
        "Assistant:",
    ]

    def __init__(self):
        self._tokenizer = None
        self._model = None

    def _ensure_model(self):
        if self._model is None or self._tokenizer is None:
            from .documind_service import documind_service
            self._model = documind_service.qwen_model
            self._tokenizer = documind_service.qwen_tokenizer

    def _resolve_token_limit(self, mode: str, max_tokens: Optional[int] = None) -> int:
        if max_tokens and max_tokens > 0:
            return max(max_tokens, 1024)

        mode_clean = mode.lower().strip()
        if mode_clean in ("summary", "summarize"):
            return 896
        elif mode_clean in ("comparison", "compare"):
            return 1024
        elif mode_clean in ("calculation", "math", "reconcile", "reconciliation"):
            return 1024
        elif mode_clean in ("enumeration", "list"):
            return 1024
        elif mode_clean in ("qa", "factual", "explain", "explanation"):
            return 1024
        elif mode_clean in ("classification", "extraction"):
            return 1024
        return 1024

    def format_chatml(
        self,
        question: str,
        context_text: str,
        structured_context: Optional[str] = None,
    ) -> str:
        """Constructs Qwen ChatML template using tokenizer or explicit ChatML markup."""
        self._ensure_model()

        user_content_parts = []
        if structured_context:
            user_content_parts.append(f"### VERIFIED STRUCTURED DATA & COMPUTATIONS:\n{structured_context}")

        user_content_parts.append(f"### DOCUMENT CONTEXT:\n{context_text}")
        user_content_parts.append(f"### USER QUESTION:\n{question}")
        user_content_parts.append("Directly answer the question above in Sentence 1. Provide only factual details explicitly verified by the evidence.")

        full_user_content = "\n\n".join(user_content_parts)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": full_user_content},
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logger.debug("apply_chat_template fallback to manual ChatML: %s", e)

        # Fallback to standard ChatML format
        return (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{full_user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def sanitize_output(self, raw_text: str) -> str:
        """
        Deterministic post-generation sanitation:
        - Strips ChatML artifacts (<|im_start|>, <|im_end|>, etc.)
        - Truncates hallucinated conversational role-turns (Human:, User:, Assistant:)
        - Strips pleasantries, filler, apologies, and meta-notes
        """
        cleaned = raw_text

        # 1. Truncate at first role turn hallucination
        for stop_seq in ["\nHuman:", "\nUser:", "\nAssistant:", "Human:", "User:", "Assistant:"]:
            if stop_seq in cleaned:
                cleaned = cleaned.split(stop_seq)[0]

        # 2. Strip ChatML tags
        cleaned = re.sub(r"<\|im_start\|>.*?(?:\n|$)", "", cleaned)
        cleaned = re.sub(r"<\|im_end\|>", "", cleaned)
        cleaned = re.sub(r"<\|endoftext\|>", "", cleaned)

        # 3. Strip trailing pleasantries and meta notes
        cleaned = re.sub(r"\s*(?:I hope this helps!?|Hope this helps!?|Let me know if you have any other questions!?|Have a great day!?).*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n*(?:Note:[^\n]*)$", "", cleaned, flags=re.IGNORECASE)

        # 4. Iteratively strip compound introductory filler & procedural preambles
        preamble_patterns = [
            r"^Certainly!?:?",
            r"^Sure!?:?",
            r"^Here is[^\n]*?:?",
            r"^Below is[^\n]*?:?",
            r"^Based on the (?:provided|context)[^\n]*?:?",
            r"^According to the (?:provided|context|document)[^\n]*?:?",
            r"^To answer your query[^\n]*?:?",
            r"^Step \d+:\s*(?:Let us begin|Extract|Begin)[^\n]*?(?:\n|\.|\:|$)",
            r"^Let us (?:begin|analyze)[^\n]*?(?:\n|\.|\:|$)",
            r"^I apologize[^\n]*?\.\s*",
            r"^I am sorry[^\n]*?\.\s*",
            r"^I would be happy to help[^\n]*?\n*",
            r"^In this document,\s*",
            r"^Looking at the context provided,\s*",
            r"^To address (?:the|your) query(?: regarding)?[^\n]*?:?",
            r"^To answer (?:the|your) query(?: regarding)?[^\n]*?:?",
        ]
        changed = True
        while changed:
            changed = False
            for pat in preamble_patterns:
                m = re.match(pat, cleaned.strip(), flags=re.IGNORECASE)
                if m:
                    cleaned = cleaned.strip()[m.end():]
                    changed = True

        return cleaned.strip()

    def reason(
        self,
        question: str,
        context_text: str,
        structured_context: Optional[str] = None,
        mode: str = "qa",
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Synthesizes grounded reasoning response using Qwen2.5-1.5B:
        - ChatML formatted prompt
        - StopSequenceCriteria
        - Near-deterministic temperature (0.05)
        - Repetition penalty (1.08)
        - Output sanitization
        """
        # Hard context & grounding gatekeeper: immediately return negative restraint without touching model
        if not context_text or not context_text.strip():
            return "Not Mentioned in Provided Context"

        self._ensure_model()

        prompt = self.format_chatml(
            question=question,
            context_text=context_text,
            structured_context=structured_context,
        )

        device = next(self._model.parameters()).device
        inputs = self._tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = inputs["input_ids"].shape[1]

        token_budget = self._resolve_token_limit(mode, max_tokens)

        # Configure stopping criteria
        stop_criteria = StopSequenceCriteria(
            tokenizer=self._tokenizer,
            stop_strings=self.STOP_STRINGS,
            prompt_length=prompt_length,
        )
        criteria_list = StoppingCriteriaList([stop_criteria])

        try:
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=token_budget,
                    temperature=0.05,
                    top_p=0.90,
                    repetition_penalty=1.08,
                    do_sample=False,  # Near-deterministic greedy decoding
                    stopping_criteria=criteria_list,
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            gen_ids = outputs[0, prompt_length:]
            raw_answer = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
            sanitized = self.sanitize_output(raw_answer)
            return sanitized if sanitized else "Not Mentioned in Provided Context"

        except Exception as e:
            logger.exception("Grounded reasoning inference failed: %s", e)
            return "Not Mentioned in Provided Context"


reasoning_engine = GroundedReasoningEngine()
