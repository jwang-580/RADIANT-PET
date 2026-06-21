import json
import re
import os
from tqdm import tqdm
from dotenv import load_dotenv

# ==========================================
# Setup & Helper Functions
# ==========================================

def system_prompt(use_report=False):
    if use_report:
        return (
        f"You are provided with a json string of a PET/CT–derived description of a site of FDG uptake in a patient with lymphoma. You are also provided with the radiology report of the same patient.\n"
        f"Based on the json description of the lesion and the report, consider where the uptake is located and whether it represents physiological activity or lymphoma.\n"
    )
    else:
        return (
        f"You are provided with a PET/CT–derived description of a site of FDG uptake in a patient with lymphoma.\n"
        f"Consider where the uptake is located and whether it represents physiological activity or lymphoma.\n"
    )

def format_lesion_input(lesion_data):
    """
    Formats the lesion dictionary into a string representation similar to the notebook.
    Removes 'id', 'symmetric_partner_id', 'is_true_lesion', 'overlap_ratio' and other metadata.
    """
    d = lesion_data.copy()
    # Remove metadata keys that aren't part of the physical description
    keys_to_remove = ['id', 'symmetric_partner_id', 'is_true_lesion', 'overlap_ratio']
    for k in keys_to_remove:
        if k in d:
            del d[k]
    # Also remove predicted fields if they exist from previous runs
    if 'predicted_type' in d: del d['predicted_type']
    if 'predicted_site' in d: del d['predicted_site']
    
    return str(d)

def parse_response(decoded_text):
    """
    Extracts the classification (physiological_site or lesion_site) and the site name.
    Returns: (type_string, site_string)
    
    Handles multiple response formats:
    - "physiological_site: [site_name]" (with brackets)
    - "physiological_site: site_name" (without brackets)
    - "lesion_site: [site_name]" (with brackets)
    - "lesion_site: site_name" (without brackets)
    """
    # Look for the pattern in the entire response
    # Pattern matches:
    # - "lesion_site: [site_name]" or "physiological_site: [site_name]" (with brackets)
    # - "lesion_site: site_name" or "physiological_site: site_name" (without brackets, captures until end of line)
    matches = list(re.finditer(
        r"(physiological_site|lesion_site)\s*:\s*(?:\[\s*([^\]]+)\s*\]|([^\s\n][^\n]*))", 
        decoded_text, 
        re.IGNORECASE
    ))
    
    if matches:
        last_match = matches[-1]
        p_type = last_match.group(1).lower()
        # group(2) is content inside [], group(3) is content without []
        p_site = last_match.group(2) if last_match.group(2) else last_match.group(3)
        if p_site:
            return p_type, p_site.strip()
    
    return None, None

def get_true_lesion_ids(data):
    """
    Extracts the list of true lesion IDs from the data dictionary.
    Prioritizes 'metadata.ground_truth.true_lesion_ids'.
    """
    if 'metadata' in data and 'ground_truth' in data['metadata'] and 'true_lesion_ids' in data['metadata']['ground_truth']:
        return set(data['metadata']['ground_truth']['true_lesion_ids'])
    # Fallback or empty if not found
    return set()

# ==========================================
# Main Execution
# ==========================================

def process_file(file_path, model_type, client, canonical_sites, use_report=False, debug=False, output_dir=None):
    print(f"\nProcessing file: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if 'lesions' not in data:
        print(f"Error: 'lesions' key not found in {file_path}. Skipping.")
        return

    lesions = data['lesions']
    print(f"Loaded {len(lesions)} lesions.")
    
    # Identify true lesion IDs
    true_lesion_ids = get_true_lesion_ids(data)
    print(f"Found {len(true_lesion_ids)} true lesions from metadata.")

    # Inference Loop
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    errors = 0
    
    # Check if report is available when requested
    report_available = 'metadata' in data and 'pet_report' in data['metadata']
    if use_report and not report_available:
        print(f"Warning: --use_report flag set but 'pet_report' key not found in {file_path}. Processing without report.")
    
    for lesion in tqdm(lesions, desc=f"Processing Lesions in {os.path.basename(file_path)}"):
        lid = lesion['id']
        if use_report and report_available:
            report = data['metadata']['pet_report']
            input_desc = f"{format_lesion_input(lesion)}\n\nCorresponding Radiology Report:\n{report}"
        else:
            input_desc = format_lesion_input(lesion)

        if use_report and report_available:
            prompt_content = input_desc + "\n\n" +  "Center coords are in voxel space centered at (0,0,0), with positive values indicating left, anterior, and superior relative to the body center. The uptake location can be inferred from coords, organ overlaps, closest organs, and the vertebral level of the uptake's center" + "\n\n" + f"Available sites to choose from: {canonical_sites}" + "\n\n" + "Is this uptake physiological activity or lymphoma? final answer formatted as 'physiological_site: [site]' if physiological activity or 'lesion_site: [site]' if lymphoma; for example: 'physiological_site: small_bowel'"
        else:
            prompt_content = input_desc + "\n\n" + "Center coords are in voxel space centered at (0,0,0), with positive values indicating left, anterior, and superior relative to the body center. The uptake location can be inferred from coords, organ overlaps, closest organs, and the vertebral level of the uptake's center" + "\n\n" + f"Available sites to choose from: {canonical_sites}" + "\n\n" + "Is this uptake physiological activity or lymphoma? final answer formatted as 'physiological_site: [site]' if physiological activity or 'lesion_site: [site]' if lymphoma."
        
        if debug:
            print("\n" + "="*40)
            print("DEBUG: Prompt Content")
            print("="*40)
            print(f"SYSTEM: {system_prompt(use_report)}")
            print(f"\nUSER: {prompt_content}")
            print("="*40)
            # In debug mode, we just check the first lesion and exit the file processing
            return
        
        try:
            if model_type == "gemini":
                # Gemini API call
                full_prompt = system_prompt(use_report) + "\n\n" + prompt_content
                response = client.models.generate_content(model=client.model_name, contents=full_prompt)
                decoded_output = response.text
            elif model_type == "openai":
                # OpenAI API call
                messages = [
                    {"role": "system", "content": system_prompt(use_report)},
                    {"role": "user", "content": prompt_content}
                ]
                response = client.responses.create(
                    model=client.model_name,
                    input=messages,
                    reasoning={"effort": "low"}
                )
                decoded_output = response.output_text
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            print(f"\nError processing lesion {lid}: {e}")
            decoded_output = ""
            errors += 1
        
        # Debug: Print the model response
        print(f"\n[Lesion {lid}] Model Response:")
        print(decoded_output)
        print("-" * 40)
        
        # Parse result
        pred_type, pred_site = parse_response(decoded_output)
        
        lesion['predicted_type'] = pred_type
        lesion['predicted_site'] = pred_site
        
        if 'is_true_lesion' in lesion:
            is_true_lesion = lesion['is_true_lesion']
        else:
            is_true_lesion = lid in true_lesion_ids
            
        is_pred_lesion = (pred_type == "lesion_site")
        is_pred_physio = (pred_type == "physiological_site")
        
        if is_true_lesion:
            if is_pred_lesion:
                tp += 1
            elif is_pred_physio:
                fn += 1
            else:
                fn += 1
                errors += 1
        else:
            if is_pred_physio:
                tn += 1
            elif is_pred_lesion:
                fp += 1
            else:
                fp += 1
                errors += 1
    
    # Save to output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, os.path.basename(file_path))
    else:
        output_file = file_path.replace(".json", "_predictions.json")
    
    print(f"Saving updated results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    denom_precision = tp + fp
    precision = tp / denom_precision if denom_precision > 0 else 0
    denom_recall = tp + fn
    recall = tp / denom_recall if denom_recall > 0 else 0
    denom_f1 = precision + recall
    f1 = 2 * (precision * recall) / denom_f1 if denom_f1 > 0 else 0
    
    print("-" * 40)
    print(f"Metrics for {os.path.basename(file_path)}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}, Errors: {errors}")
    print(f"Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")
    print("-" * 40)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on PET/CT lesion descriptions using Gemini or OpenAI.")
    parser.add_argument("--data_dir", type=str, default="eval/", help="Directory containing JSON results files.")
    parser.add_argument("--model_type", type=str, choices=["gemini", "openai"], default="gemini", help="Type of model to use (gemini or openai).")
    parser.add_argument("--model_name", type=str, default="gemini-3-flash-preview", help="Model name (e.g., 'gemini-3-flash-preview' for Gemini or 'gpt-4o' for OpenAI).")
    parser.add_argument("--site_file", type=str, default="data/lymphoma_site_lists.json", help="Path to the site file.")
    parser.add_argument("--use_report", action="store_true", help="Use the radiology report in the inference.")
    parser.add_argument("--debug", action="store_true", help="Print the prompt for the first lesion and exit without running inference/saving.")
    args = parser.parse_args()
    
    print(f"Setting up Inference...")
    print(f"Model Type: {args.model_type}")
    print(f"Model Name: {args.model_name}")
    print(f"Data Directory: {args.data_dir}")
    
    # Load Site List
    try:
        all_sites = json.load(open(args.site_file))
        canonical_sites = list(set(all_sites["lymphoma_sites"] + all_sites["physiological_sites"]))
    except Exception as e:
        print(f"Warning: Could not load site list from {args.site_file}: {e}")
        canonical_sites = []
    
    client = None
    
    if not args.debug:
        # Load API keys and initialize client
        load_dotenv()
        
        if args.model_type == "gemini":
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("Error: GEMINI_API_KEY not found in .env file.")
                print("Please create a .env file with: GEMINI_API_KEY=your_api_key_here")
                return
            print("Initializing Gemini client...")
            client = genai.Client(api_key=api_key)
            client.model_name = args.model_name
            
        elif args.model_type == "openai":
            from openai import OpenAI
            api_key = os.getenv("OAI_API_KEY")
            if not api_key:
                print("Error: OPENAI_API_KEY not found in .env file.")
                print("Please create a .env file with: OPENAI_API_KEY=your_api_key_here")
                return
            print("Initializing OpenAI client...")
            client = OpenAI(api_key=api_key)
            client.model_name = args.model_name
    else:
        print("Debug mode: Skipping model loading.")
    
    # Find all JSON files in the directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} not found.")
        return
        
    all_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".json")]
    
    print(f"Found {len(all_files)} files to process.")
    
    # Create output directory based on model_name
    output_dir = f"{args.model_name.replace('/', '_')}_results"
    print(f"Results will be saved to: {output_dir}")
    
    for file_path in all_files:
        if file_path.endswith("_predictions.json"):
            continue
        
        process_file(file_path, args.model_type, client, canonical_sites, use_report=args.use_report, debug=args.debug, output_dir=output_dir)

if __name__ == "__main__":
    main()
