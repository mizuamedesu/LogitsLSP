import os
import torch
import json
import argparse
from transformers import LogitsProcessor, AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class LogitObserverProcessor(LogitsProcessor):
    def __init__(self, tokenizer, top_k=15):
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.logit_history = []
        self.generated_tokens = []
        self.previous_length = 0

    def __call__(self, input_ids, scores):
        top_scores, top_indices = torch.topk(scores[0], min(self.top_k, scores.size(1)))

        step_data = {
            "tokens": [],
            "logits": [],
            "selected": None
        }

        for i in range(len(top_scores)):
            token_id = top_indices[i].item()
            token_text = self.tokenizer.decode([token_id])
            logit_score = top_scores[i].item()

            step_data["tokens"].append(token_text)
            step_data["logits"].append(logit_score)

        if len(input_ids[0]) > self.previous_length and self.previous_length > 0:
            generated_token_id = input_ids[0][-1].item()
            generated_token_text = self.tokenizer.decode([generated_token_id])

            if len(self.logit_history) > 0:
                self.logit_history[-1]["selected"] = generated_token_text
                self.generated_tokens.append(generated_token_text)

        self.previous_length = len(input_ids[0])
        self.logit_history.append(step_data)
        return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('user_input', type=str)
    parser.add_argument('--output', type=str, default='logits.json')
    args = parser.parse_args()
    
    model_name = "Qwen/Qwen3-8B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    messages = [{"role": "user", "content": args.user_input}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    observer = LogitObserverProcessor(tokenizer, top_k=15)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=[observer]
        )

    input_length = inputs['input_ids'].shape[1]
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    if len(observer.logit_history) > 0:
        last_token_id = outputs[0][-1].item()
        last_token_text = tokenizer.decode([last_token_id])
        observer.logit_history[-1]["selected"] = last_token_text

    result = {
        "input": args.user_input,
        "response": generated_text,
        "logit_scores": observer.logit_history
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()