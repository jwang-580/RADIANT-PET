import os
import argparse
import re
import json
import difflib
import torch
from unsloth import FastModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from accelerate import Accelerator

# ==========================================
# Configuration
# ==========================================
MAX_SEQ_LENGTH = 4096
MAX_PROMPT_LENGTH = 2048
LORA_RANK = 8
MODEL_NAME = "unsloth/gpt-oss-20b-BF16"
SITE_FILE = "data/lymphoma_site_lists.json"
PUBLIC_DATASETS = {
    "hs-unet": "data/autopet_nnunet/train",
    "threshold": "data/autopet_threshold/train",
}

# ==========================================
# Helper Functions & Constants
# ==========================================

all_sites = json.load(open(SITE_FILE))
CANONICAL_SITES = list(set(all_sites["lymphoma_sites"] +
                           all_sites["physiological_sites"]))

EQUIV_GROUPS = [
    {"bone", "bone_marrow"},
    {"tonsil", "tonsillar"},
    {"axil_skeleton", "bone"},
    {"appendicular_skeleton", "bone"},
    {"parotid", "Parotid", "salivary_glands"},
    {"gi", "esophagus"},
    {"gi", "stomach"},
    {"gi", "duodenum"},
    {"gi", "small_bowel"},
    {"gi", "colon"},
    {"gi", "rectum"},
]

def in_same_equiv_group(site_a, site_b):
    if not site_a or not site_b:
        return False
    sa = site_a.lower()
    sb = site_b.lower()
    for group in EQUIV_GROUPS:
        # true if site_a has any token from group AND site_b has any token from same group
        if (any(token in sa for token in group) and
            any(token in sb for token in group)):
            return True
    return False

# ==========================================
# Prompt Formatting
# ==========================================
#REASONING_START = "<start_working_out>"
#REASONING_END   = "<end_working_out>"
#SOLUTION_START = "<SOLUTION>"
#SOLUTION_END = "</SOLUTION>"

MATCH_FORMAT = re.compile(
    r"assistantfinal(.*)",
    re.DOTALL
)

def system_prompt():
    return (
        f"You are provided with a PET/CT–derived description of a site of FDG uptake in a patient with lymphoma.\n"
        f"Consider where the uptake is located and whether it represents physiological activity or lymphoma.\n"
    )

# ==========================================
# Reward Functions
# ==========================================

# def match_format_exactly(completions, **kwargs):
#     scores = []
#     for completion in completions:
#         score = 0.0
#         response = completion[0]["content"]
#         if MATCH_FORMAT.search(response) is not None:
#             score += 3.0
#         scores.append(score)
#     return scores

def check_answer(prompts, completions, answer, **kwargs):
    responses = [completion[0]["content"] for completion in completions]

    # Debug: Print first response to see what model is doing
    if len(responses) > 0:
        print(f"\n[DEBUG] Model Response:\n{responses[0]}\n[DEBUG] End Response\n")

    extracted_responses = []
    for r in responses:
        match = MATCH_FORMAT.search(r)
        if match:
            extracted_responses.append(match.group(1))
        else:
            extracted_responses.append(None)

    def parse_type_and_site(s):
        if not s: return None, None
        m = re.match(
            r"\s*(physiological_site|lesion_site)\s*:\s*(?:\[\s*([^\]]+)\s*\]|([^\n]+))\s*$",
            s,
            flags=re.IGNORECASE
        )
        if m:
            prefix = m.group(1).lower()
            site   = m.group(2) if m.group(2) else m.group(3)
            return prefix, site.strip()
        return None, None

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0.0
        if guess is None:
            scores.append(0.0)
            continue

        guess_type, guess_site = parse_type_and_site(guess)
        true_type, true_site   = parse_type_and_site(true_answer)

        if true_type and guess_type:
            if guess_type == true_type:
                score += 1
            # if both physiological_site and lesion_site are present, score -1
            elif ("physiological_site" in guess.lower()) and ("lesion_site" in guess.lower()):
                scores.append(-1.0)
                continue
        
        if true_site:
            if guess_site == true_site:
                score += 2
            elif in_same_equiv_group(guess_site, true_site):
                score += 1.5
        scores.append(score)
    return scores

def no_cheating(completions, **kwargs):
    """Penalize if the solution is just copied from the prompt or reasoning without work? 
    For now, just a placeholder or simple check."""
    # Placeholder: maybe check if reasoning is empty?
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        # If reasoning is too short, maybe penalize?
        # For now, return 0
        scores.append(0.0)
    return scores

def check_reasoning(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    scores = []
    for r in responses:
        if "suv max" in r.lower() or "max suv" in r.lower() or "suv_max" in r.lower() or "max_suv" in r.lower():
            scores.append(0.0)
        else:
            scores.append(-0.5)
    return scores

# ==========================================
# Main Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Train the report-free RADIANT-PET GRPO adapter.")
    parser.add_argument(
        "--candidate-source",
        choices=sorted(PUBLIC_DATASETS),
        default="hs-unet",
        help="AutoPET candidate-generation method used to build the training set",
    )
    parser.add_argument("--data-dir", default=None, help="Optional Hugging Face Dataset/DatasetDict path")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data_file = args.data_dir or PUBLIC_DATASETS[args.candidate_source]
    output_dir = args.output_dir or f"outputs/grpo_autopet_{args.candidate_source}"

    # accelerator = Accelerator()
    # local_rank = accelerator.local_process_index  # 0, 1, ... per process

    # device_map = {"": local_rank}

    # 1. Load Model
    print("Loading model...")
    model, tokenizer = FastModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = MAX_SEQ_LENGTH, 
        load_in_4bit = False,
        full_finetuning = False, 
        # device_map = device_map,
        dtype = torch.bfloat16,
        # offload_embeddings = True,
    )

    model = FastModel.get_peft_model(
        model,
        # finetune_vision_layers     = False,
        # finetune_language_layers   = True,
        # finetune_attention_modules = True,
        # finetune_mlp_modules       = True,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        r = LORA_RANK,
        lora_alpha = 2 * LORA_RANK, 
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 2. Load Dataset
    print("Loading dataset...")
    from datasets import load_from_disk
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Training dataset not found: {data_file}")
    dataset = load_from_disk(data_file)

    # Convert to HuggingFace Dataset
    # Our data has 'input_text' (lesion desc) and 'output_text' (ground truth).
    # We need to format it for the model.
    
    def format_example(item):
        prompt = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": item["input_text"] + "\n\n" + "Center coords are in voxel space centered at (0,0,0), with positive values indicating left, anterior, and superior relative to the body center. The uptake location can be inferred from coords, organ overlaps, closest organs, and the vertebral level of the uptake’s center" + "\n\n" + f"Available sites to choose from: {CANONICAL_SITES}" + "\n\n" + "Is this uptake physiological activity or lymphoma? final answer formatted as 'physiological_site: [site]' if physiological activity or 'lesion_site: [site]' if lymphoma."}
            #{"role": "user", "content": item["input_text"] + "\n\n" + "Center coords are in voxel space centered at (0,0,0), with positive values indicating left, anterior, and superior relative to the body center. The uptake location can be inferred from coords, organ overlaps, closest organs, and the vertebral level of the uptake’s center" + "\n\n" + "Is this uptake physiological activity or lymphoma? final answer formatted as 'physiological_site: [site]' if physiological activity or 'lesion_site: [site]' if lymphoma; for example: 'physiological_site: small_bowel'."}
        ]
        return {
            "prompt": prompt,
            "answer": item["output_text"]
        }

    dataset = dataset.map(format_example)
    
    train_dataset = dataset["train"] if hasattr(dataset, "keys") and "train" in dataset else dataset
    # eval_dataset = dataset["validation"] if "validation" in dataset else None

    # 3. Training Config
    training_args = GRPOConfig(
        temperature = 1.0,
        learning_rate = 5e-5,
        weight_decay = 0.001,
        warmup_ratio = 0.1,
        lr_scheduler_type = "linear",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        num_generations = 4,
        max_prompt_length = MAX_PROMPT_LENGTH, 
        max_completion_length = MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH,
        max_steps = 1200,
        output_dir = output_dir,
        save_strategy = "steps",
        save_steps = 200,
        # === REQUIRED FOR DDP, per Unsloth docs ===
        # ddp_find_unused_parameters = False,
    )

    # 4. Trainer
    print("Initializing trainer...")
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            no_cheating,
            check_answer,
            check_reasoning,
        ],
        args = training_args,
        train_dataset = train_dataset
    )

    # 5. Train
    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # 6. Save Model
    print("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
