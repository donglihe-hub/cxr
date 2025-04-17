import tensorflow as tf
import tensorflow_text
import os

# (Optional) Limit to CPU/GPU use — useful if running on TPU machine
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # or "" for CPU only

# Enable resource variables (fixes RefVariable warnings)
tf.compat.v1.enable_resource_variables()

# Step 2: Define a wrapper module with an exportable function
class ExportModule(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name="input_example")
    ])
    def serve(self, input_example):
        # Call the original serving_default signature
        return self.model.signatures["serving_default"](input_example)

def main(model_dir, export_dir):
    # Step 1: Load the original SavedModel
    model = tf.saved_model.load(model_dir)

    # Step 3: Instantiate the exportable module
    export_model = ExportModule(model)

    # Step 4: Re-save the model with CPU/GPU-compatible ops only
    tf.saved_model.save(
        export_model,
        export_dir=export_dir,
        signatures={"serving_default": export_model.serve}
    )

    print(f"Model re-exported to {export_dir} with CPU/GPU ops only.")

if __name__ == "__main__":
    model_dir = '.tf_models/hf/elixr-c-v2-pooled'
    export_dir = ".tf_models/hf/elixr-c-v2-pooled_gpu"
    main(model_dir, export_dir)