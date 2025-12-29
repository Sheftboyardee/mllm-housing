"""
Generate natural language descriptions for houses using Qwen2-VL model.
Processes 4 images per house (bathroom, bedroom, kitchen, frontal) and generates descriptions.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
import torch

# Global variable for max generation tokens (can be set via command line)
MAX_GENERATION_TOKENS = 160


def load_associations(associations_path: str) -> List[Dict[str, Any]]:
    """Load house associations from JSON file."""
    with open(associations_path, 'r', encoding='utf-8') as f:
        associations = json.load(f)
    return associations


def load_existing_results(output_path: str) -> tuple[List[Dict[str, Any]], set[int]]:
    """
    Load existing results from output file if it exists.
    
    Args:
        output_path: Path to the output JSON file
    
    Returns:
        Tuple of (existing_results_list, set_of_processed_house_ids)
    """
    if not os.path.exists(output_path):
        return [], set()
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
        
        # Extract processed house IDs
        processed_ids = {result.get("house_id") for result in existing_results if "house_id" in result}
        
        return existing_results, processed_ids
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Warning: Could not load existing results from {output_path}: {e}")
        print("  Starting fresh...")
        return [], set()


def load_qwen_vl_model(model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
    """Load Qwen2-VL model, tokenizer, and processor."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA available! Using GPU: {gpu_name}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        dtype = torch.float16
    else:
        device = "cpu"
        print("Using CPU")
        dtype = torch.float32
    
    try:
        # Check if accelerate is available
        import accelerate  # noqa: F401
        print("Using device_map='auto' with accelerate...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype
        )
        # Verify model is on GPU
        if device == "cuda":
            param_device = next(model.parameters()).device
            if param_device.type == "cuda":
                print(f"Model loaded on GPU: {param_device}")
            else:
                print(f"Warning: Model loaded on {param_device}, expected GPU")
    except (ImportError, ValueError) as e:
        if "accelerate" in str(e).lower() or isinstance(e, ImportError):
            print("Warning: accelerate not available, loading model to device manually...")
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=dtype
            )
            model = model.to(device)
            print(f"Model moved to {device}")
        else:
            raise
    
    print("Model loaded successfully!")
    return model, tokenizer, processor


def load_image(image_path: str, base_path: str = ".", max_size: int = 512) -> Image.Image:
    """
    Load and resize an image from the given path.
    
    Args:
        image_path: Path to the image
        base_path: Base directory path
        max_size: Maximum dimension (width or height) to resize to. Smaller = faster processing.
    
    Returns:
        Resized PIL Image in RGB format
    """

    filename = os.path.basename(image_path)
    full_path = os.path.join(base_path, "data", "Houses-dataset", "Houses Dataset", filename)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")
    
    # Load and convert to RGB
    img = Image.open(full_path).convert("RGB")
    
    # Resize if image is larger than max_size
    if max_size > 0:
        width, height = img.size
        if width > max_size or height > max_size:
            # Maintain aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return img


def generate_description_for_house(
    model,
    tokenizer,
    house: Dict[str, Any],
    base_path: str = ".",
    processor=None,
    max_image_size: int = 512
) -> str:
    """
    Generate a natural language description for a house from its 4 images.
    
    Args:
        model: Qwen2-VL
        tokenizer: The tokenizer for the model
        house: House dictionary with images and metadata
        base_path: Base path for resolving image paths
    
    Returns:
        Generated description string
    """
    images_dict = house["images"]
    
    # Load all 4 images in order: frontal, bedroom, bathroom, kitchen
    image_order = ["frontal", "bedroom", "bathroom", "kitchen"]
    images = []
    loaded_types = []
    
    for img_type in image_order:
        if img_type in images_dict:
            try:
                img = load_image(images_dict[img_type], base_path, max_size=max_image_size)
                images.append(img)
                loaded_types.append(img_type)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
    
    if not images:
        return f"No images found for house {house['house_id']}"
    
    
    prompt_text = (
        "Based on the provided images of this house, write a detailed, natural description. "
        "Include information about:\n"
        "- The exterior appearance and architectural style (from the frontal view)\n"
        "- The bedroom(s) - size, layout, furnishings, and condition\n"
        "- The bathroom(s) - style, fixtures, and amenities\n"
        "- The kitchen - design, appliances, and features\n"
        "- Overall condition, quality, and any notable features\n"
        "Write a comprehensive, natural description that would help someone visualize this property, given a 160 token limit."
    )
    
    # Prepare messages format for Qwen2-VL
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt_text})
    
    messages = [{"role": "user", "content": content}]
    
    # Process vision information using qwen_vl_utils
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Apply chat template to get text
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Use processor to prepare inputs (processor handles the image processing)
    if processor is not None:
        # Processor expects the processed image inputs from process_vision_info
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
    else:
        # Fallback: use tokenizer only (won't work properly with images)
        inputs = tokenizer(text, padding=True, return_tensors="pt")
    
    # Move inputs to model device
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    
    # inference_mode is faster than no_grad
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=MAX_GENERATION_TOKENS,
            do_sample=False,  # Greedy decoding is faster
            num_beams=1  # No beam search for speed
        )
        # Extract only the generated tokens (remove input tokens)
        if "input_ids" in inputs:
            input_length = inputs["input_ids"].shape[1]
            generated_ids_trimmed = generated_ids[:, input_length:]
        else:
            generated_ids_trimmed = generated_ids
        
        # Decode using processor if available, otherwise tokenizer
        decoder = processor if processor is not None else tokenizer
        output_text = decoder.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    
    return output_text[0] if output_text else "Failed to generate description"


def save_results(results: List[Dict[str, Any]], output_path: str, is_final: bool = False):
    """
    Save results to JSON file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the file
        is_final: Whether this is the final save (affects backup behavior)
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Backup existing file only on first save (not on incremental saves)
    if is_final and os.path.exists(output_path):
        import shutil
        from datetime import datetime
        backup_path = output_path.replace('.json', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        shutil.copy2(output_path, backup_path)
        print(f"Backed up previous file to: {backup_path}")
    
    # Write results (overwrites existing file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    if is_final:
        print(f"\nSaved descriptions to {output_path}")
        print(f"Processed {len(results)} houses successfully")
    else:
        print(f"Progress saved: {len(results)} houses processed so far")


def process_all_houses(
    associations: List[Dict[str, Any]],
    model,
    tokenizer,
    base_path: str = ".",
    output_path: str = "data/Houses-dataset/house_descriptions.json",
    processor=None,
    max_image_size: int = 512,
    save_interval: int = 10,
    resume: bool = True
):
    """
    Process all houses and generate descriptions.
    
    Args:
        associations: List of house associations
        model: Qwen2-VL 
        tokenizer: Model tokenizer
        base_path: Base path for resolving image paths
        output_path: Path to save generated descriptions
        processor: Model processor
        max_image_size: Maximum image dimension
        save_interval: Save results every N houses (default: 10)
        resume: If True, skip already processed houses from existing output file
    """
    # Load existing results if resuming
    if resume:
        existing_results, processed_ids = load_existing_results(output_path)
        if existing_results:
            print(f"Found existing results: {len(existing_results)} houses already processed")
            print(f"Processed house IDs: {sorted(list(processed_ids))[:10]}{'...' if len(processed_ids) > 10 else ''}")
            print(f"Resuming from where we left off...\n")
            results = existing_results.copy()
        else:
            results = []
            processed_ids = set()
    else:
        results = []
        processed_ids = set()
    
    # Filter out already processed houses
    houses_to_process = [
        house for house in associations 
        if house.get("house_id") not in processed_ids
    ]
    
    total = len(associations)
    remaining = len(houses_to_process)
    already_done = len(processed_ids)
    
    if remaining == 0:
        print(f"All {total} houses have already been processed!")
        print(f"  Results are in: {output_path}")
        return
    
    print(f"Processing {total} houses total:")
    print(f"Already processed: {already_done}")
    print(f"Remaining to process: {remaining}")
    print(f"Results will be saved every {save_interval} houses and on completion.")
    print("Press Ctrl+C to stop early - current progress will be saved.\n")
    
    try:
        for idx, house in enumerate(houses_to_process, 1):
            house_id = house["house_id"]
            overall_idx = already_done + idx
            print(f"\nProcessing house {house_id} ({overall_idx}/{total}, {idx}/{remaining} remaining)...")
            
            try:
                description = generate_description_for_house(
                    model, tokenizer, house, base_path, processor, max_image_size
                )
                
                result = {
                    "house_id": house_id,
                    "metadata": house.get("metadata", {}),
                    "description": description,
                    "images": house.get("images", {})
                }
                results.append(result)
                
                print(f"Generated description for house {house_id}")
                
            except Exception as e:
                print(f"Error processing house {house_id}: {str(e)}")
                result = {
                    "house_id": house_id,
                    "metadata": house.get("metadata", {}),
                    "description": f"Error: {str(e)}",
                    "images": house.get("images", {})
                }
                results.append(result)
            
            # Save incrementally every N houses
            if idx % save_interval == 0:
                save_results(results, output_path, is_final=False)
        
        # Final save
        save_results(results, output_path, is_final=True)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
        print(f"Saving {len(results)} houses processed so far...")
        save_results(results, output_path, is_final=True)
        print(f"\nPartial results saved to {output_path}")
        print(f"  Run the script again to resume from where you left off.")
        raise  # Re-raise to exit gracefully


def main():
    """Main function to run the description generation pipeline."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate house descriptions from images")
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=384,
        help="Maximum image dimension (width/height) in pixels. Smaller = faster. Default: 384"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=160,
        help="Maximum tokens to generate. Default: 160"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save results every N houses. Default: 10"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing results - start fresh"
    )
    args = parser.parse_args()
    
    # Paths
    base_path = Path(__file__).parent.parent
    associations_path = base_path / "data" / "Houses-dataset" / "associations.json"
    output_path = base_path / "data" / "Houses-dataset" / "house_descriptions.json"
    
    # Load associations
    print("Loading associations...")
    associations = load_associations(str(associations_path))
    print(f"Loaded {len(associations)} houses")
    print(f"Optimization settings: max_image_size={args.max_image_size}, max_tokens={args.max_tokens}")
    
    # Load model
    model, tokenizer, processor = load_qwen_vl_model()
    
    # Update max_tokens in the generation function
    global MAX_GENERATION_TOKENS
    MAX_GENERATION_TOKENS = args.max_tokens
    
    # Process all houses (resume by default unless --no-resume is specified)
    process_all_houses(
        associations,
        model,
        tokenizer,
        base_path=str(base_path),
        output_path=str(output_path),
        processor=processor,
        max_image_size=args.max_image_size,
        save_interval=args.save_interval,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()

