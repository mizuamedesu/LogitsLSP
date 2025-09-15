#!/usr/bin/env python3
import os
import sys
import torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
from lsp_client import LSPClient
from lsp_manager import LSPManager
from lsp_logits_processor import LSPAwareLogitsProcessor
from Observation.LogitsObservation import LogitObserverProcessor

warnings.filterwarnings("ignore")
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


LSP_SERVERS = {
    "python": ["pylsp"],
    "typescript": ["typescript-language-server", "--stdio"],
    "javascript": ["typescript-language-server", "--stdio"],
    "rust": ["rust-analyzer"],
    "cpp": ["clangd"],
    "c": ["clangd"],
    "go": ["gopls"],
    "java": ["jdtls"]
}


class CombinedLogitsProcessor:
    def __init__(self, processors):
        self.processors = processors

    def __call__(self, input_ids, scores):
        for processor in self.processors:
            scores = processor(input_ids, scores)
        return scores


def main():
    if len(sys.argv) < 3:
        print("Usage: python LogitsLSP_Simple.py <prompt> <language>")
        print("Example: python LogitsLSP_Simple.py \"Write hello world\" rust")
        print(f"Supported languages: {', '.join(LSP_SERVERS.keys())}")
        sys.exit(1)

    prompt = sys.argv[1]
    language = sys.argv[2].lower()

    if language not in LSP_SERVERS:
        print(f"Error: Unsupported language '{language}'")
        print(f"Supported languages: {', '.join(LSP_SERVERS.keys())}")
        sys.exit(1)

    output_file = sys.argv[3] if len(sys.argv) > 3 else f"{language}_output.json"
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 256


    model_name = "Qwen/Qwen3-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    workspace = os.getcwd()

    lsp_manager = LSPManager(workspace_path=workspace, auto_install=True)
    lsp_client = lsp_manager.get_client(language=language)

    if not lsp_client:
        print(f"Error: Failed to start LSP server for {language}")
        config = lsp_manager.LSP_CONFIGS.get(language, {})
        if config and config.get("servers"):
            server = config["servers"][0]
            print(f"Please install the LSP server manually:")
            print(f"  {server['install_cmd']}")
        sys.exit(1)

    file_extensions = {
        "python": ".py",
        "typescript": ".ts",
        "javascript": ".js",
        "rust": ".rs",
        "cpp": ".cpp",
        "c": ".c",
        "go": ".go",
        "java": ".java"
    }
    temp_file = f"/tmp/temp{file_extensions.get(language, '.txt')}"
    lsp_client.open_document(temp_file, "")

    processors = []

    observer = LogitObserverProcessor(tokenizer, top_k=15)
    processors.append(observer)

    lsp_processor = LSPAwareLogitsProcessor(
        tokenizer,
        lsp_client,
        mask_strength=-10.0,
        use_adaptive_mask=True,
        lookahead_tokens=3,
        use_completion_scores=True
    )
    processors.append(lsp_processor)

    combined_processor = CombinedLogitsProcessor(processors)

    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=8192)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=[combined_processor]
        )

    input_length = inputs['input_ids'].shape[1]
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    if len(observer.logit_history) > 0:
        last_token_id = outputs[0][-1].item()
        last_token_text = tokenizer.decode([last_token_id])
        observer.logit_history[-1]["selected"] = last_token_text

    result = {
        "prompt": prompt,
        "language": language,
        "response": generated_text,
        "logit_scores": observer.logit_history
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(generated_text)

    lsp_manager.shutdown_all()


if __name__ == "__main__":
    main()