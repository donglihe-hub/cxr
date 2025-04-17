import io
import os
import png
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import tensorflow_text
from clientside.clients import make_hugging_face_client
from pydicom.pixels import pixel_array
from tqdm import tqdm
from huggingface_hub import snapshot_download
from PIL import Image

_MODEL_DIR = ".tf_models/hf"
snapshot_download(repo_id="google/cxr-foundation",local_dir=_MODEL_DIR,
                  allow_patterns=['elixr-c-v2-pooled/*', 'pax-elixr-b-text/*'])

_BERT_TF_HUB_PATH = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

# Helper function for processing image data
def png_to_tfexample(image_array: np.ndarray) -> tf.train.Example:
    """Creates a tf.train.Example from a NumPy array."""
    # Convert the image to float32 and shift the minimum value to zero
    image = image_array.astype(np.float32)
    image -= image.min()

    if image_array.dtype == np.uint8:
        # For uint8 images, no rescaling is needed
        pixel_array = image.astype(np.uint8)
        bitdepth = 8
    else:
        # For other data types, scale image to use the full 16-bit range
        max_val = image.max()
        if max_val > 0:
            image *= 65535 / max_val  # Scale to 16-bit range
        pixel_array = image.astype(np.uint16)
        bitdepth = 16

    # Ensure the array is 2-D (grayscale image)
    if pixel_array.ndim != 2:
        raise ValueError(f'Array must be 2-D. Actual dimensions: {pixel_array.ndim}')

    # Encode the array as a PNG image
    output = io.BytesIO()
    png.Writer(
        width=pixel_array.shape[1],
        height=pixel_array.shape[0],
        greyscale=True,
        bitdepth=bitdepth
    ).write(output, pixel_array.tolist())
    png_bytes = output.getvalue()

    # Create a tf.train.Example and assign the features
    example = tf.train.Example()
    features = example.features.feature
    features['image/encoded'].bytes_list.value.append(png_bytes)
    features['image/format'].bytes_list.value.append(b'png')

    return example

def generate_embeddings(data_dir):
    data_paths = list(data_dir.glob("**/*.dcm"))

    Path(data_paths[0].parent.parent / "embeddings").mkdir(parents=True, exist_ok=True)

    print("Available GPUs:", tf.config.list_physical_devices('GPU'))
    cxr_client = make_hugging_face_client(_MODEL_DIR)
    
    for data_path in tqdm(data_paths):
        img = Image.fromarray(pixel_array(data_path))
        # with tf.device('/GPU:0'):
        embeddings = cxr_client.get_image_embeddings_from_images([img])[0]
        embeddings_general = np.array(embeddings.general_img_emb)
        embeddings_contrastive = np.array(embeddings.contrastive_img_emb)
        print(data_path.parent / f"{data_path.stem}_general.npy")
        np.save(data_path.parent / f"{data_path.stem}_general.npy", embeddings_general)
        np.save(data_path.parent / f"{data_path.stem}_contrastive.npy", embeddings_contrastive)


if __name__ == "__main__":
    data_dir = Path("data/XR_DICOM")
    generate_embeddings(data_dir)
