# inspect_qwen.py

import torch
from models import Encoder # Assuming 'models.py' and 'Encoder' class exist

def print_tree(module, f, prefix=""):
    # print direct children only
    for name, child in module.named_children():
        print(f"{prefix}{name} -> {child.__class__.__name__}", file=f)
        print_tree(child, f, prefix + name + ".")

if __name__ == "__main__":
    output_filename = "qwen_model_inspection.txt"

    # Open the file in write mode
    with open(output_filename, "w") as f:
        # 1) instantiate your Encoder (on CPU is fine)
        # Note: This assumes you have the 'models.py' file and the Encoder
        # class defined correctly. If not, this part will raise an error.
        try:
            enc = Encoder(head="MLP")
            # 2) grab the raw Qwen2_5_VLForConditionalGeneration
            qwen_wrap = enc.vlm.qwen
            print("=== wrapper layers (qwen) ===", file=f)
            print_tree(qwen_wrap, f)

            # 3) it has an attribute `.model` which is the inner Qwen2_5_VLModel
            qwen_model = qwen_wrap.model
            print("\n=== inner Qwen2_5_VLModel layers ===", file=f)
            print_tree(qwen_model, f)

            # 4) that has .language_model → the Qwen2_5_VLTextModel
            text_mod = qwen_model.language_model
            print("\n=== text_model (Qwen2_5_VLTextModel) layers ===", file=f)
            print_tree(text_mod, f)

            print(f"\nOutput successfully written to {output_filename}")
        except ImportError:
            error_message = (
                "Error: The 'models' module or 'Encoder' class could not be "
                "imported. Please ensure 'models.py' exists in the same "
                "directory and is correctly defined."
            )
            print(error_message, file=f)
            print(error_message) # Also print to console for immediate feedback
        except AttributeError as e:
            error_message = (
                f"AttributeError: {e}. This might be due to an unexpected "
                "model structure or an issue with the 'Encoder' instantiation."
            )
            print(error_message, file=f)
            print(error_message) # Also print to console
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            print(error_message, file=f)
            print(error_message) # Also print to console