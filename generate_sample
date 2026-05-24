import os
import re
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

DEFAULT_MODEL_PATH = ""
DEFAULT_MATH_DATA_PATH = ""
DEFAULT_OUTPUT_DIR = ""

MAX_LENGTH = 1024
MAX_NEW_TOKENS = 768
BASE_PROB_THRESHOLD = 0.2
TOP_K_CANDIDATES = 6
sample_num = 5
BETA = 10


def clean_latex_symbols(s):
    if s is None or s == "":
        return ""
    s = re.sub(r'\$', '', s)
    s = re.sub(r'\\left|\\right', '', s)
    return s.replace(" ", "").strip()


def extract_math_numeric_answer(answer_str):
    try:
        pattern = r'\\boxed\{(?:[^{}]|\{[^{}]*\})*\}'
        matches = re.findall(pattern, answer_str)
        if not matches:
            return ""
        boxed_content = matches[-1].replace("\\boxed{", "")
        if boxed_content.endswith('}'):
            boxed_content = boxed_content[:-1]
        return boxed_content
    except:
        return ""


def build_prompts(tokenizer, example):
    instruction = """You are given a grade school math question. Please answer the question in the following format:

        Q: <Question>
        A: <Think step by step here> \\boxed{{<number only answer>}}

        Format requirements : you must first output your reasoning before finalized with the "\\boxed{{}}" format followed by the final numeric answer

Reference Answer:
{reference_answer}

Now answer the question yourself:
"""
    instruction = instruction.format(reference_answer=example['solution'])
    ref_human_content = instruction + f"Q: {example['problem'].strip()}\nA: "
    ref_messages = [{"role": "user", "content": ref_human_content}]
    ref_text = tokenizer.apply_chat_template(
        ref_messages,
        tokenize=False,
        add_generation_prompt=True,
        system_message=""
    )

    instruction = """You are given a grade school math question. Please answer the question in the following format:

    Q: <Question>
    A: <Think step by step here> \\boxed{{<number only answer>}}

    Format requirements : you must first output your reasoning before finalized with the "\\boxed{{}}" format followed by the final numeric answer
    """
    base_human_content = instruction + f"Q: {example['problem'].strip()}\nA: "
    base_messages = [{"role": "user", "content": base_human_content}]
    base_text = tokenizer.apply_chat_template(
        base_messages,
        tokenize=False,
        add_generation_prompt=True,
        system_message=""
    )

    return base_text, ref_text


def constrained_generate(model, tokenizer, base_input_ids, ref_input_ids, device, verbose=True):
    generated_ids = base_input_ids.clone()
    current_ref_ids = ref_input_ids.clone()

    base_cache = None
    ref_cache = None

    full_decoded_text = ""

    for step in range(MAX_NEW_TOKENS):
        with torch.inference_mode():
            base_input_slice = generated_ids[:, -1:] if base_cache is not None else generated_ids

            base_outputs = model(
                input_ids=base_input_slice,
                past_key_values=base_cache,
                use_cache=True
            )
            base_logits = base_outputs.logits[:, -1, :]
            base_cache = base_outputs.past_key_values
            base_probs = torch.softmax(base_logits, dim=-1)

            ref_input_slice = current_ref_ids[:, -1:] if ref_cache is not None else current_ref_ids
            ref_outputs = model(
                input_ids=ref_input_slice,
                past_key_values=ref_cache,
                use_cache=True
            )
            ref_logits = ref_outputs.logits[:, -1, :]
            ref_cache = ref_outputs.past_key_values
            ref_probs = torch.softmax(ref_logits, dim=-1)

        top_k_values, top_k_indices = torch.topk(ref_probs, k=TOP_K_CANDIDATES, dim=-1)

        selected_token_id = None
        max_prob_in_candidates = -1.0
        best_candidate_id_in_topk = None

        for i in range(TOP_K_CANDIDATES):
            candidate_id = top_k_indices[0, i].item()

            candidate_base_prob = base_probs[0, candidate_id].item()

            if candidate_base_prob > max_prob_in_candidates:
                max_prob_in_candidates = candidate_base_prob
                best_candidate_id_in_topk = candidate_id

            if candidate_base_prob >= BASE_PROB_THRESHOLD:
                selected_token_id = candidate_id
                decision_type = "Ref proposal accepted"
                break

        if selected_token_id is None:
            selected_token_id = best_candidate_id_in_topk
            decision_type = "Select best candidate"

        if selected_token_id == tokenizer.eos_token_id:
            break

        next_token_tensor = torch.tensor([[selected_token_id]], device=device)
        generated_ids = torch.cat([generated_ids, next_token_tensor], dim=-1)
        current_ref_ids = torch.cat([current_ref_ids, next_token_tensor], dim=-1)

    return generated_ids


def hd_generate(model, tokenizer, base_input_ids, ref_input_ids, device, beta=BETA, verbose=False):
    generated_ids = base_input_ids.clone()
    current_ref_ids = ref_input_ids.clone()

    base_cache = None
    ref_cache = None

    vocab_size = model.config.vocab_size
    log_vocab = torch.log(torch.tensor(float(vocab_size), device=device))

    for step in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            base_outputs = model(
                input_ids=generated_ids[:, -1:] if base_cache else generated_ids,
                past_key_values=base_cache,
                use_cache=True
            )
            base_logits = base_outputs.logits[:, -1, :]
            base_cache = base_outputs.past_key_values

            ref_outputs = model(
                input_ids=current_ref_ids[:, -1:] if ref_cache else current_ref_ids,
                past_key_values=ref_cache,
                use_cache=True
            )
            ref_logits = ref_outputs.logits[:, -1, :]
            ref_cache = ref_outputs.past_key_values

            ref_probs = torch.softmax(ref_logits, dim=-1)
            entropy = -torch.sum(ref_probs * torch.log(ref_probs + 1e-10), dim=-1)

            log_ref = F.log_softmax(ref_logits, dim=-1)
            entropy = -torch.sum(log_ref.exp().double() * log_ref.double(), dim=-1).float()

            norm_entropy = entropy / log_vocab

            lam = torch.clamp(beta * norm_entropy, 0.0, 1.0)

            log_ref = F.log_softmax(ref_logits, dim=-1)
            log_base = F.log_softmax(base_logits, dim=-1)

            log_mix = (1.0 - lam) * log_ref + lam * log_base

            probs_mix = torch.exp(log_mix)
            next_token_id = torch.multinomial(probs_mix, num_samples=1).squeeze(1)

            selected_token_id = next_token_id.item()

            if selected_token_id == tokenizer.eos_token_id:
                break

            next_token_tensor = torch.tensor([[selected_token_id]], device=device)
            generated_ids = torch.cat([generated_ids, next_token_tensor], dim=-1)
            current_ref_ids = torch.cat([current_ref_ids, next_token_tensor], dim=-1)

    return generated_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_path", type=str, default=DEFAULT_MATH_DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample_num", type=int, default=sample_num)
    parser.add_argument("--threshold", type=float, default=BASE_PROB_THRESHOLD)
    parser.add_argument("--data_mode", type=str, default='cd', choices=['cd', 'hd'])
    parser.add_argument("--decode_mode", type=str, default='base', choices=['base', 'ref'])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    device = model.device

    dataset = load_dataset(args.data_path, split="train")
    dataset = dataset.select(range(min(args.sample_num, len(dataset))))

    results = []

    for example in tqdm(dataset, desc="Generating"):
        base_text, ref_text = build_prompts(tokenizer, example)

        base_inputs = tokenizer(base_text, return_tensors="pt", truncation=True, padding=False, max_length=MAX_LENGTH,
                                add_special_tokens=False).to(device)
        ref_inputs = tokenizer(ref_text, return_tensors="pt", truncation=True, padding=False, max_length=MAX_LENGTH,
                               add_special_tokens=False).to(device)

        if args.data_mode == 'cd':
            output_ids = constrained_generate(model, tokenizer, base_inputs.input_ids, ref_inputs.input_ids, device)
        elif args.data_mode == 'hd':
            output_ids = hd_generate(model, tokenizer, base_inputs.input_ids, ref_inputs.input_ids, device)

        prompt_len = base_inputs.input_ids.shape[1]
        gen_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)

        pred_num = extract_math_numeric_answer(gen_text)
        gold_num = example['answer']

        results.append({
            "problem": example['problem'],
            "generated_answer": gen_text,
            "pred_numeric": pred_num,
            "gold_numeric": gold_num,
            "is_correct": pred_num == gold_num,
            "level": example['level']
        })

    correct_count = sum(1 for r in results if r["is_correct"])
    acc = correct_count / len(results) * 100

    os.makedirs(args.output_dir, exist_ok=True)

    final_dataset = Dataset.from_list(results)

    consistent_dataset = final_dataset.filter(lambda x: x["is_correct"], desc="Filtering correct answers")
    inconsistent_dataset = final_dataset.filter(lambda x: not x["is_correct"], desc="Filtering wrong answers")

    if args.data_mode == 'cd':
        folder_name = f"{args.model_path.split('/')[-2]}_contrastive_sample{args.sample_num}_{args.threshold}_{args.decode_mode}_cd"
    else:
        folder_name = f"{args.model_path.split('/')[-2]}_contrastive_sample{args.sample_num}_{BETA}_{args.data_mode}_hd"
    save_dir = os.path.join(args.output_dir, folder_name)

    filter_dir = save_dir + "_correct"
    infilter_dir = save_dir + "_incorrect"

    os.makedirs(args.output_dir, exist_ok=True)

    if len(consistent_dataset) > 0:
        consistent_dataset.save_to_disk(filter_dir)

    if len(inconsistent_dataset) > 0:
        inconsistent_dataset.save_to_disk(infilter_dir)

    acc = (len(consistent_dataset) / len(final_dataset)) * 100 if len(final_dataset) > 0 else 0


if __name__ == "__main__":
    main()
