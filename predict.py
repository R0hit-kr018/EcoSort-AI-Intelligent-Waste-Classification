"""
Prediction script for EcoSort AI - Waste Classification

This script provides command-line inference capabilities for the EcoSort AI model.
It handles cross-platform file paths, validates model existence, and provides
detailed error messages for troubleshooting.
"""

from ultralytics import YOLO
from pathlib import Path
import sys

# ============================================================================
# PATH CONFIGURATION - Cross-platform file handling with pathlib
# ============================================================================

# Get the absolute path to the predict.py file
SCRIPT_DIR = Path(__file__).parent.resolve()
# Project root is the same as script directory
PROJECT_ROOT = SCRIPT_DIR.resolve()
# Construct model path relative to project root
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

# Debug: Print paths for troubleshooting
print("[DEBUG] ========================================")
print(f"[DEBUG] SCRIPT_DIR: {SCRIPT_DIR}")
print(f"[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}")
print(f"[DEBUG] MODEL_PATH: {MODEL_PATH}")
print(f"[DEBUG] Model exists: {MODEL_PATH.exists()}")
print("[DEBUG] ========================================\n")


def validate_model() -> bool:
    """
    Validate that the model file exists and is readable.
    
    Returns:
        bool: True if model is valid, False otherwise
    """
    if not MODEL_PATH.exists():
        print(f"❌ [ERROR] Model file not found at: {MODEL_PATH}")
        print(f"[ERROR] Absolute path: {MODEL_PATH.resolve()}")
        print("\n[HELP] To fix this issue:")
        print("  1. Train the model: python train.py")
        print("  2. Or copy best.pt from runs/classify/train-4/weights/ to models/")
        print("  3. Verify the file exists: python -c \"from pathlib import Path; print(Path('models/best.pt').exists())\"")
        return False
    
    if not MODEL_PATH.is_file():
        print(f"❌ [ERROR] Model path exists but is not a file: {MODEL_PATH}")
        return False
    
    # Check file size
    file_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    if file_size_mb < 1:
        print(f"❌ [ERROR] Model file is too small ({file_size_mb:.2f}MB). File may be corrupted.")
        return False
    
    print(f"✅ [OK] Model file validated")
    print(f"   Location: {MODEL_PATH}")
    print(f"   Size: {file_size_mb:.2f}MB\n")
    return True


def load_model() -> YOLO:
    """
    Load the YOLO model with comprehensive error handling.
    
    Returns:
        YOLO: Loaded model instance
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    try:
        print(f"📦 Loading model from: {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
        print("✅ Model loaded successfully!\n")
        return model
    except FileNotFoundError as e:
        print(f"❌ [ERROR] Model file not found: {e}")
        raise
    except Exception as e:
        print(f"❌ [ERROR] Failed to load model: {e}")
        raise


def predict(image_path: str) -> dict:
    """
    Run inference on an image using the EcoSort AI model.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Dictionary containing prediction results with keys:
            - 'category': Predicted waste category
            - 'confidence': Confidence score (0-1)
            - 'percentage': Confidence as percentage
            - 'all_predictions': Dict of all category predictions
    """
    # Validate image path
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        print(f"❌ [ERROR] Image file not found: {image_path}")
        print(f"[ERROR] Absolute path: {image_path_obj.resolve()}")
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if not image_path_obj.is_file():
        print(f"❌ [ERROR] Image path is not a file: {image_path}")
        raise ValueError(f"Not a file: {image_path}")
    
    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    if image_path_obj.suffix.lower() not in valid_extensions:
        print(f"❌ [ERROR] Invalid image format: {image_path_obj.suffix}")
        print(f"[ERROR] Supported formats: {', '.join(valid_extensions)}")
        raise ValueError(f"Invalid image format: {image_path_obj.suffix}")
    
    try:
        # Load model
        model = load_model()
        
        # Run prediction
        print(f"🔍 Running inference on: {image_path}")
        results = model.predict(str(image_path), verbose=False)
        
        # Extract prediction results
        top1_idx = int(results[0].probs.top1)
        top1_confidence = float(results[0].probs.top1conf)
        predicted_label = results[0].names[top1_idx]
        
        # Get all predictions
        all_predictions = {}
        for i, prob in enumerate(results[0].probs.data):
            category_name = results[0].names[i]
            prob_value = float(prob)
            all_predictions[category_name] = f"{prob_value*100:.2f}%"
        
        # Build result dictionary
        result = {
            'category': predicted_label,
            'confidence': top1_confidence,
            'percentage': f"{top1_confidence*100:.2f}%",
            'all_predictions': all_predictions
        }
        
        return result
        
    except Exception as e:
        print(f"❌ [ERROR] Prediction failed: {e}")
        raise


def main():
    """Main entry point for the prediction script."""
    
    print("\n" + "="*60)
    print("🌍 EcoSort AI - Waste Classification Prediction")
    print("="*60 + "\n")
    
    # Validate model
    if not validate_model():
        sys.exit(1)
    
    # Use test image from dataset
    test_image = PROJECT_ROOT / "dataset" / "val" / "plastic" / "plastic5.jpg"
    
    # Alternative: Accept command-line argument
    if len(sys.argv) > 1:
        test_image = Path(sys.argv[1])
    
    try:
        # Run prediction
        result = predict(str(test_image))
        
        # Display results
        print("📊 PREDICTION RESULTS")
        print("-" * 60)
        print(f"✅ Predicted Waste Category: {result['category'].upper()}")
        print(f"   Confidence: {result['percentage']}")
        print(f"   Confidence Score: {result['confidence']:.4f}")
        print("\n📈 All Predictions:")
        for category, probability in result['all_predictions'].items():
            print(f"   • {category}: {probability}")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
