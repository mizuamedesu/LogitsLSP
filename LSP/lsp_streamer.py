import sys
import time
from typing import List, Optional
from transformers import TextStreamer


class LSPAwareStreamer(TextStreamer):

    def __init__(self, tokenizer, lsp_processor, skip_prompt=True, skip_special_tokens=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
        self.lsp_processor = lsp_processor
        self.last_suggestions = []
        self.suggestion_display_active = False
        self.selected_suggestion = None

    def on_finalized_text(self, text: str, stream_end: bool = False):
        current_suggestions = self.lsp_processor.get_current_suggestions() if self.lsp_processor else []
        if current_suggestions and current_suggestions != self.last_suggestions:
            self._show_suggestions(current_suggestions, text)
            self.last_suggestions = current_suggestions
            self.suggestion_display_active = True
        elif self.suggestion_display_active and not current_suggestions:
            self._clear_suggestions()
            self.suggestion_display_active = False
        if not self.suggestion_display_active:
            sys.stdout.write(text)
            sys.stdout.flush()
        else:
            self._show_selected_text(text)

    def _show_suggestions(self, suggestions: List[str], current_text: str):
        sys.stdout.write('\033[s')
        sys.stdout.write(current_text)

        sys.stdout.write('\033[90m')
        sys.stdout.write(' [LSP: ')

        for i, suggestion in enumerate(suggestions[:5]):
            if i > 0:
                sys.stdout.write(' | ')
            display_text = suggestion[:20] + '...' if len(suggestion) > 20 else suggestion
            sys.stdout.write(f'{i+1}.{display_text}')

        sys.stdout.write(']')
        sys.stdout.write('\033[0m')
        sys.stdout.write('\033[u')
        sys.stdout.flush()

    def _clear_suggestions(self):
        sys.stdout.write('\033[K')
        sys.stdout.flush()

    def _show_selected_text(self, text: str):
        self._clear_suggestions()
        sys.stdout.write(text)
        sys.stdout.flush()


class InteractiveLSPStreamer(TextStreamer):

    def __init__(self, tokenizer, lsp_processor, skip_prompt=True, skip_special_tokens=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
        self.lsp_processor = lsp_processor
        self.suggestion_box_shown = False
        self.current_line = ""
        self.suggestion_start_pos = 0

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.current_line += text
        suggestions = self.lsp_processor.get_current_suggestions() if self.lsp_processor else []

        if suggestions and not self.suggestion_box_shown:
            self._show_suggestion_box(suggestions)
            self.suggestion_box_shown = True
            self.suggestion_start_pos = len(self.current_line)

        elif not suggestions and self.suggestion_box_shown:
            self._hide_suggestion_box()
            self.suggestion_box_shown = False
        if '\n' in text:
            self.current_line = ""
            if self.suggestion_box_shown:
                self._hide_suggestion_box()
                self.suggestion_box_shown = False
        sys.stdout.write(text)
        sys.stdout.flush()

    def _show_suggestion_box(self, suggestions: List[str]):
        sys.stdout.write('\033[s')
        sys.stdout.write('\n')
        sys.stdout.write('\033[48;5;236m')
        sys.stdout.write('┌─ LSP Suggestions ─────────────┐\n')

        for i, suggestion in enumerate(suggestions[:5]):
            if i == 0:
                sys.stdout.write('│ \033[48;5;24m▶ ')
            else:
                sys.stdout.write('│   ')

            display_text = suggestion[:27] + '...' if len(suggestion) > 27 else suggestion
            sys.stdout.write(f'{display_text:<30}')

            if i == 0:
                sys.stdout.write('\033[48;5;236m')
            sys.stdout.write('│\n')

        sys.stdout.write('└────────────────────────────────┘')
        sys.stdout.write('\033[0m')
        sys.stdout.write('\033[u')
        sys.stdout.flush()

    def _hide_suggestion_box(self):
        sys.stdout.write('\033[s')

        for _ in range(7):
            sys.stdout.write('\033[B\033[K')
        sys.stdout.write('\033[u')
        sys.stdout.flush()