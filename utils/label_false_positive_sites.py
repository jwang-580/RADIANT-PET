import pandas as pd
import json
import os
import glob
import argparse
from google import genai
import dotenv
from tqdm import tqdm

# Load environment variables
dotenv.load_dotenv()

def generate_fp_training_data(
    json_folder="nnunet/nnunet_raw/dataset001_autopet/labelstr/nnunet_t8/descriptions_with_gt",
    site_list_path="data/lymphoma_site_lists.json",
    output_file="training_autopet_nnunet_fp_data.jsonl",
    model_name="gemini-3-flash-preview",
    limit=None,
    include_report=True
):
    # Initialize Gemini client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        return

    client = genai.Client(api_key=api_key)
    
    # Load physiological sites
    try:
        with open(site_list_path, 'r') as f:
            site_data = json.load(f)
            all_physio_sites = site_data.get("physiological_sites", [])
            if not all_physio_sites:
                print("Warning: 'physiological_sites' not found or empty in site list file.")
    except Exception as e:
        print(f"Error loading site list file: {e}")
        return

    # Get list of JSON files
    json_files = (
    glob.glob(os.path.join(json_folder, "*_results.json")) +
    glob.glob(os.path.join(json_folder, "*_description.json"))
)
    print(f"Found {len(json_files)} JSON files.")
    
    if limit:
        json_files = json_files[:limit]
        print(f"Limiting to first {limit} files.")

    results = []

    for json_file_path in tqdm(json_files, desc="Processing files"):
        try:
            # Extract case_id from filename
            filename = os.path.basename(json_file_path)
            # Assuming filename format: {case_id}_results.json
            if filename.endswith("_results.json"):
                case_id = filename.removesuffix("_results.json")
            elif filename.endswith("_description.json"):
                case_id = filename.removesuffix("_description.json")
            else:
                raise ValueError(f"Unexpected filename format: {filename}")
            
            # Load JSON file
            with open(json_file_path, 'r') as f:
                result_data = json.load(f)
            
            if 'lesions' not in result_data:
                continue

            for lesion in result_data['lesions']:
                # Check if this is a true positive (skip if it is)
                is_true_lesion = lesion.get("is_true_lesion", None)
                if is_true_lesion is True:
                    continue
                
                # Check volume (consistent with generate_gt.py)
                if lesion.get('volume_voxels', 0) < 20:
                    continue

                # Get radiology report if available and requested
                radiology_report = ""
                if include_report:
                    radiology_report = result_data.get('metadata', {}).get('pet_report', '')
                
                # Construct prompt based on whether report is included
                if include_report and radiology_report:
                    prompt = f"""
A PET/CT–derived description of a physiological uptake is provided below:
{lesion}

The radiology report for this case is provided below:
{radiology_report}

The uptake is known to belong to one of the following predefined physiological update sites:
{all_physio_sites}

Using the anatomical coordinates, adjacent structures, and organ relationships described, determine which site most accurately corresponds to this uptake. 
Return only the name of the site, "physiological_site: <site_name>".
If unsure or site not listed, return "physiological_site: unknown".
"""
                else:
                    prompt = f"""
A PET/CT–derived description of a physiological uptake is provided below:
{lesion}

The uptake is known to belong to one of the following predefined physiological update sites:
{all_physio_sites}

Using the anatomical coordinates, adjacent structures, and organ relationships described, determine which site most accurately corresponds to this uptake. 
Return only the name of the site, "physiological_site: <site_name>".
If unsure or site not listed, return "physiological_site: unknown".
"""
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    
                    output_text = response.text.strip()
                    
                    # Extract specific fields for input_text
                    keys_to_keep = [
                        "center_coords", "volume_voxels", "max_suv", "mean_suv", 
                        "organ_overlaps", "closest_organs", "vertebrae_level", 
                        "shape", "is_symmetric", "symmetry_score"
                    ]
                    filtered_lesion = {k: lesion.get(k) for k in keys_to_keep}
                    
                    # Append to results
                    training_example = {
                        "input_text": f"{filtered_lesion}",
                        "report": radiology_report,
                        "output_text": output_text,
                        "case_id": case_id,
                        "lesion_id": lesion.get("id")
                    }
                    results.append(training_example)
                    
                    with open(output_file, 'a') as out_f:
                        out_f.write(json.dumps(training_example) + "\n")
                        
                except Exception as e:
                    print(f"Error generating content for lesion in {case_id}: {e}")
                    continue

        except Exception as e:
            print(f"Error processing file {json_file_path}: {e}")
            continue

    print(f"Processing complete. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training data for physiological uptake site classification using LLM"
    )
    
    parser.add_argument(
        "--json-folder",
        type=str,
        default="nnunet/nnunet_raw/dataset001_autopet/labelstr/nnunet_t8/descriptions_with_gt",
        help="Path to folder containing JSON files with lesion descriptions"
    )
    
    parser.add_argument(
        "--site-list-path",
        type=str,
        default="data/lymphoma_site_lists.json",
        help="Path to JSON file containing physiological site lists"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="training_autopet_nnunet_fp_data.jsonl",
        help="Path to output JSONL file"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemini-3-flash-preview",
        help="Name of the Gemini model to use"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of files to process (default: process all)"
    )
    
    parser.add_argument(
        "--include-report",
        action="store_true",
        default=True,
        help="Include radiology report in the prompt (default: True)"
    )
    
    parser.add_argument(
        "--no-report",
        action="store_false",
        dest="include_report",
        help="Exclude radiology report from the prompt"
    )
    
    args = parser.parse_args()
    
    generate_fp_training_data(
        json_folder=args.json_folder,
        site_list_path=args.site_list_path,
        output_file=args.output_file,
        model_name=args.model_name,
        limit=args.limit,
        include_report=args.include_report
    )

