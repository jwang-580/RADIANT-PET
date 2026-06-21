import json
import re
import os
import glob
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Configuration
TP_FILE = "training_autopet_nnunet_tp_data.jsonl"
FP_FILE = "training_autopet_nnunet_fp_data.jsonl"
SITE_LIST_FILE = "data/lymphoma_site_lists.json"
OUTPUT_DIR = "data/dataset/autopet_nnunet"

# Distribution settings
BALANCE_SITES = True  # Balance examples across sites
TOTAL_CASES = 8000    # Maximum total cases (None = use all with balancing)


def load_site_lists(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Combine all sites into a single set for validation
        valid_sites = set()
        if "lymphoma_sites" in data:
            valid_sites.update(data["lymphoma_sites"])
        if "physiological_sites" in data:
            valid_sites.update(data["physiological_sites"])
            
        return valid_sites
    except Exception as e:
        print(f"Error loading site lists: {e}")
        return set()

def clean_and_validate(text, valid_sites):
    """
    Extracts 'physiological_site: [site]' or 'lesion_site: [site]'
    and validates if [site] is in valid_sites.
    """
    if not text:
        return None
        
    # Regex to capture the full answer line
    # Matches: physiological_site: [liver] OR physiological_site: liver
    # Group 1: prefix
    # Group 2: site (inside brackets)
    # Group 3: site (no brackets)
    match = re.search(
        r"(physiological_site|lesion_site)\s*:\s*(?:\[\s*([^\]]+)\s*\]|([^\n]+))", 
        text, 
        flags=re.IGNORECASE
    )
    
    if match:
        prefix = match.group(1).lower() # physiological_site or lesion_site
        site = match.group(2) if match.group(2) else match.group(3)
        site = site.strip()
        
        # Validate site
        # We check if the extracted site matches one in the list (case-insensitive?)
        # The lists in JSON are mixed case (e.g. "Parotid", "tonsil_left").
        # Let's try exact match first, then case-insensitive match against the set.
        
        # Create a lowercase map for validation
        valid_sites_lower = {s.lower() for s in valid_sites}
        
        if site in valid_sites or site.lower() in valid_sites_lower:
            # Reconstruct the clean output
            # User requested: "only need to extract 'physiological_site...' and 'lesion_site:...'"
            # I will standardize format to use brackets as per DPRO.py expectations?
            # The prompt in DPRO.py asks for brackets.
            return f"{prefix}: [{site}]"
        else:
            # print(f"Skipping invalid site: {site}")
            return None
            
    return None

def distribute_cases(all_examples, total_cases=None, balance_sites=True):
    """
    Distribute cases to ensure equal representation across sites.
    
    Args:
        all_examples: List of example dictionaries with 'output_text' containing site labels
        total_cases: Maximum total number of cases to select (None = use all)
        balance_sites: If True, balance distribution across sites
        
    Returns:
        List of selected examples with balanced site distribution
    """
    if not all_examples:
        return []
    
    if not balance_sites:
        # Just limit total if specified
        if total_cases:
            return all_examples[:total_cases]
        return all_examples
    
    # Extract site from each example
    # output_text format: "physiological_site: [liver]" or "physiological_site: liver" or "lesion_site: bone"
    site_examples = {}
    
    for example in all_examples:
        output_text = example.get('output_text', '')
        # Extract site name - handle both bracketed and non-bracketed formats
        # Match: "physiological_site: [liver]" or "physiological_site: liver"
        match = re.search(r'(physiological_site|lesion_site)\s*:\s*(?:\[([^\]]+)\]|([^\n,]+))', output_text, re.IGNORECASE)
        if match:
            site = (match.group(2) or match.group(3)).strip()
            if site not in site_examples:
                site_examples[site] = []
            site_examples[site].append(example)
    
    if not site_examples:
        print("Warning: No sites found in examples")
        return all_examples[:total_cases] if total_cases else all_examples
    
    # Calculate distribution
    num_sites = len(site_examples)
    print(f"Found {num_sites} unique sites")
    
    if total_cases:
        # Distribute evenly across sites
        per_site = total_cases // num_sites
        remainder = total_cases % num_sites
        print(f"Target: {total_cases} total cases, ~{per_site} per site")
    else:
        # Use minimum count across all sites to balance
        per_site = min(len(examples) for examples in site_examples.values())
        print(f"Balancing to {per_site} cases per site (minimum available)")
    
    # Select examples from each site
    selected = []
    site_counts = {}
    
    for site, examples in sorted(site_examples.items()):
        # Take per_site examples from this site
        count = per_site
        # Distribute remainder to first few sites
        if total_cases and remainder > 0:
            count += 1
            remainder -= 1
        
        selected_from_site = examples[:count]
        selected.extend(selected_from_site)
        site_counts[site] = len(selected_from_site)
    
    # Print distribution
    print(f"\nSite distribution:")
    for site, count in sorted(site_counts.items()):
        print(f"  {site}: {count}")
    print(f"Total selected: {len(selected)}")
    
    return selected


def main():
    # 1. Load Site Lists
    print(f"Loading site lists from {SITE_LIST_FILE}...")
    valid_sites = load_site_lists(SITE_LIST_FILE)
    print(f"Loaded {len(valid_sites)} valid sites.")
    
    # 2. Process Files
    data_files = [TP_FILE, FP_FILE]
    all_examples = []
    
    for file_path in data_files:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            continue
            
        print(f"Processing {file_path}...")
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    input_text = item.get("input_text")
                    report = item.get("report", "")  # Get report if available
                    raw_output = item.get("output_text")
                    
                    if not input_text or not raw_output:
                        continue
                    
                    # Combine input_text with report if report exists
                    if report:
                        combined_input = f"{input_text}\n\nCorresponding Radiology Report:\n{report}"
                    else:
                        combined_input = input_text
                        
                    clean_output = clean_and_validate(raw_output, valid_sites)
                    
                    if clean_output:
                        all_examples.append({
                            "input_text": combined_input,
                            "output_text": clean_output,
                            "original_output": raw_output # Optional: keep original for debug
                        })
                except json.JSONDecodeError:
                    continue
                    
    print(f"Total valid examples collected: {len(all_examples)}")
    
    if not all_examples:
        print("No valid data found. Exiting.")
        return

    # 3. Distribute cases (balance across sites if enabled)
    print(f"\nApplying case distribution...")
    all_examples = distribute_cases(all_examples, total_cases=TOTAL_CASES, balance_sites=BALANCE_SITES)
    
    if not all_examples:
        print("No examples after distribution. Exiting.")
        return

    # 4. Create Dataset
    # Split 80/20
    train_data, val_data = train_test_split(all_examples, test_size=0.02, random_state=42)
    
    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")
    
    # Create Hugging Face Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    from datasets import DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    # 5. Save
    print(f"Saving dataset to {OUTPUT_DIR}...")
    dataset_dict.save_to_disk(OUTPUT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
