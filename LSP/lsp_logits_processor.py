import torch
import math
from transformers import LogitsProcessor
from typing import List, Optional, Dict, Any, Union, Tuple
from collections import defaultdict
from lsp_manager import LSPManager
from lsp_client import LSPClient


class LSPAwareLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, lsp_input: Union[LSPClient, LSPManager],
                 temperature: float = 1.0, mask_strength: float = -10.0,
                 language: str = None, file_path: str = None,
                 use_adaptive_mask: bool = True, lookahead_tokens: int = 3,
                 use_completion_scores: bool = True):
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.base_mask_strength = mask_strength
        self.mask_strength = mask_strength
        self.current_document = ""
        self.line_number = 0
        self.char_position = 0
        self.language = language
        self.file_path = file_path
        self.use_adaptive_mask = use_adaptive_mask
        self.lookahead_tokens = lookahead_tokens
        self.use_completion_scores = use_completion_scores

        self.completion_cache = {}
        self.cache_position = None

        self.current_completion_context = ""
        self.completion_prefix_map = defaultdict(list)

        if isinstance(lsp_input, LSPManager):
            self.lsp_manager = lsp_input
            self.lsp_client = None
        else:
            self.lsp_manager = None
            self.lsp_client = lsp_input

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = scores.shape[0]

        for batch_idx in range(batch_size):
            generated_text = self.tokenizer.decode(input_ids[batch_idx], skip_special_tokens=True)
            self.current_document = generated_text
            self._update_position(generated_text)

            if self.lsp_manager:
                if not self.language:
                    self.language = self.lsp_manager.detect_language(generated_text, self.file_path)

                if self.language:
                    self.lsp_client = self.lsp_manager.get_client(
                        language=self.language,
                        code=generated_text,
                        file_path=self.file_path
                    )

            if self.lsp_client and self.lsp_client.initialized:
                if not self.lsp_client.document_uri and self.file_path:
                    self.lsp_client.open_document(self.file_path, generated_text)
                elif self.lsp_client.document_uri:
                    self.lsp_client.update_document(generated_text)

                cache_key = (self.line_number, self.char_position)
                if cache_key == self.cache_position and self.completion_cache:
                    completions = self.completion_cache
                else:
                    completions = self.lsp_client.get_completions(self.line_number, self.char_position)
                    self.completion_cache = completions
                    self.cache_position = cache_key

                if completions:
                    token_weights = self._get_weighted_tokens_from_completions(
                        completions, generated_text
                    )

                    if token_weights:
                        if self.use_adaptive_mask:
                            self._adjust_mask_strength(scores[batch_idx])

                        scores[batch_idx] = self._apply_weighted_mask(
                            scores[batch_idx], token_weights
                        )

        return scores

    def _update_position(self, text: str):
        lines = text.split('\n')
        self.line_number = len(lines) - 1
        self.char_position = len(lines[-1]) if lines else 0

    def _get_weighted_tokens_from_completions(
        self, completions: List[Dict[str, Any]], current_text: str
    ) -> Dict[int, float]:
        token_weights = {}
        self.completion_prefix_map.clear()

        for idx, completion in enumerate(completions):
            text = ""
            score = 1.0

            if isinstance(completion, dict):
                text = completion.get("label", "") or completion.get("insertText", "")
                if self.use_completion_scores:
                    sort_text = completion.get("sortText", "")
                    if sort_text:
                        score = 1.0 / (1 + len(sort_text) * 0.1)
                    elif "score" in completion:
                        score = completion.get("score", 1.0)
                    else:
                        score = 1.0 - (idx * 0.05)
            elif isinstance(completion, str):
                text = completion
                score = 1.0 - (idx * 0.05)

            if text:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)

                if tokens:
                    self.completion_prefix_map[tokens[0]].append(tokens[:self.lookahead_tokens])

                    if tokens[0] not in token_weights or token_weights[tokens[0]] < score:
                        token_weights[tokens[0]] = score

                    if len(tokens) > 1 and self.current_completion_context:
                        self._check_continuation_tokens(token_weights, tokens, score)

        return token_weights

    def _check_continuation_tokens(
        self, token_weights: Dict[int, float], tokens: List[int], score: float
    ):
        pass

    def _apply_weighted_mask(
        self, scores: torch.FloatTensor, token_weights: Dict[int, float]
    ) -> torch.FloatTensor:
        if not token_weights:
            return scores

        mask = torch.full_like(scores, self.mask_strength)

        max_weight = max(token_weights.values()) if token_weights else 1.0

        for token_id, weight in token_weights.items():
            if token_id < len(scores):
                normalized_weight = weight / max_weight
                mask_value = self.mask_strength * (1 - normalized_weight)
                mask[token_id] = mask_value

        if self.temperature != 1.0:
            scores = scores / self.temperature

        masked_scores = scores + mask

        return masked_scores

    def _adjust_mask_strength(self, scores: torch.FloatTensor):
        if not self.use_adaptive_mask:
            return

        top_k_scores, _ = torch.topk(scores, min(10, len(scores)))
        score_std = torch.std(top_k_scores).item()
        score_range = (top_k_scores[0] - top_k_scores[-1]).item()

        if score_range < 2.0:
            self.mask_strength = self.base_mask_strength * 1.5
        elif score_range > 10.0:
            self.mask_strength = self.base_mask_strength * 0.7
        else:
            self.mask_strength = self.base_mask_strength

    def update_document(self, document: str):
        self.current_document = document
        self._update_position(document)

        if abs(len(document) - len(self.current_document)) > 10:
            self.completion_cache.clear()
            self.cache_position = None

        if self.lsp_client and self.lsp_client.document_uri:
            self.lsp_client.update_document(document)