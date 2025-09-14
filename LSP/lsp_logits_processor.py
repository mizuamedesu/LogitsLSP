import torch
from transformers import LogitsProcessor
from typing import List, Optional, Dict, Any, Union
from lsp_manager import LSPManager
from lsp_client import LSPClient


class LSPAwareLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, lsp_input: Union[LSPClient, LSPManager],
                 temperature: float = 1.0, mask_strength: float = -10.0,
                 language: str = None, file_path: str = None):
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.mask_strength = mask_strength
        self.current_document = ""
        self.line_number = 0
        self.char_position = 0
        self.language = language
        self.file_path = file_path

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

                completions = self.lsp_client.get_completions(self.line_number, self.char_position)

                if completions:
                    valid_tokens = self._get_valid_tokens_from_completions(completions)

                    if valid_tokens:
                        scores[batch_idx] = self._apply_lsp_mask(scores[batch_idx], valid_tokens)

        return scores

    def _update_position(self, text: str):
        lines = text.split('\n')
        self.line_number = len(lines) - 1
        self.char_position = len(lines[-1]) if lines else 0

    def _get_valid_tokens_from_completions(self, completions: List[Dict[str, Any]]) -> List[int]:
        valid_tokens = []

        for completion in completions:
            text = ""
            if isinstance(completion, dict):
                text = completion.get("label", "") or completion.get("insertText", "")
            elif isinstance(completion, str):
                text = completion

            if text:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)

                if tokens:
                    valid_tokens.append(tokens[0])

        return list(set(valid_tokens))

    def _apply_lsp_mask(self, scores: torch.FloatTensor, valid_tokens: List[int]) -> torch.FloatTensor:
        if not valid_tokens:
            return scores

        mask = torch.full_like(scores, self.mask_strength)

        for token_id in valid_tokens:
            if token_id < len(scores):
                mask[token_id] = 0

        masked_scores = scores + mask

        return masked_scores

    def update_document(self, document: str):
        self.current_document = document
        self._update_position(document)

        if self.lsp_client and self.lsp_client.document_uri:
            self.lsp_client.update_document(document)