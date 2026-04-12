"""
SigLIP Image Encoder Module

This module provides SigLIP-based image embeddings for enhanced visual similarity scoring.
SigLIP (Sigmoid Loss for Language Image Pre-training) offers improved performance over CLIP
for image-text and image-image similarity tasks.

Uses CPU-friendly model with Apache 2.0 license: google/siglip-base-patch16-224
"""

import os
import numpy as np
import cv2
from typing import Optional, Tuple, Any
import logging

try:
    from transformers import AutoModel, AutoImageProcessor
    import torch
    SIGLIP_AVAILABLE = True
except Exception:
    SIGLIP_AVAILABLE = False
    logging.warning("SigLIP dependencies not available. Install with: pip install transformers torch")

class SigLIPEncoder:
    """
    SigLIP-based image encoder for computing visual embeddings.
    
    Features:
    - CPU-optimized inference
    - Consistent embedding dimensionality (768 for base model)
    - Normalized embeddings for cosine similarity
    - Automatic image preprocessing and resizing
    """
    
    def __init__(self, model_name: str = r"C:\Workspace\Test Automation\Label Verification\Label Verification_3\siglip-base-p16-224", device: str = "cpu"):
        """
        Initialize SigLIP encoder.
        
        Args:
            model_name: HuggingFace model identifier (default: google/siglip-base-patch16-224)
            device: Device for inference ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.embedding_dim = 768  # SigLIP base model embedding dimension
        
        if not SIGLIP_AVAILABLE:
            logging.error("SigLIP dependencies not available. Please install: pip install transformers torch")
            return
            
        try:
            self._load_model()
        except Exception as e:
            logging.error(f"Failed to load SigLIP model: {e}")
            
    def _load_model(self):
        """Load SigLIP model and processor. Prefer local model directory when present."""
        try:
            # Only load from a local path. Explicitly disallow online/hub downloads.
            try:
                is_local_dir = os.path.isdir(self.model_name)
            except Exception:
                is_local_dir = False

            try:
                is_local_file = os.path.isfile(self.model_name)
            except Exception:
                is_local_file = False

            if is_local_dir or is_local_file:
                logging.info(f"Loading SigLIP model from local path: {self.model_name}")
                # Use local_files_only to avoid network access
                self.model = AutoModel.from_pretrained(self.model_name, local_files_only=True)
                self.processor = AutoImageProcessor.from_pretrained(self.model_name, local_files_only=True)
            else:
                logging.error(
                    "SigLIP model path does not exist locally and online loading is disabled. "
                    f"Requested path: {self.model_name}"
                )
                self.model = None
                self.processor = None
                return

            # Set to evaluation mode and move to device
            self.model.eval()
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            else:
                self.device = "cpu"
                self.model = self.model.to("cpu")

            logging.info(f"SigLIP model loaded: {self.model_name} on {self.device}")

            # Diagnostic logging: report model class and available image entrypoints
            try:
                model_cls = self.model.__class__
                has_vision = hasattr(self.model, 'vision_model')
                has_get_image = hasattr(self.model, 'get_image_features')
                proc_cls = self.processor.__class__ if self.processor is not None else None
                logging.info(f"SigLIP model class: {model_cls}, vision_model: {has_vision}, get_image_features: {has_get_image}, processor: {proc_cls}")
            except Exception:
                # Non-fatal diagnostic failure
                logging.debug("SigLIP: diagnostic introspection failed")

        except Exception as e:
            logging.error(f"Error loading SigLIP model: {e}")
            self.model = None
            self.processor = None
            
    def is_available(self) -> bool:
        """Check if SigLIP encoder is available and loaded."""
        return SIGLIP_AVAILABLE and self.model is not None and self.processor is not None
        
    def preprocess_image(self, image: np.ndarray) -> Optional[Any]:
        """
        Preprocess image for SigLIP inference.
        
        Args:
            image: Input image as BGR numpy array
            
        Returns:
            Preprocessed image tensor or None if preprocessing fails
        """
        if not self.is_available():
            return None
            
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Convert grayscale to RGB by repeating channels
                if len(image.shape) == 2:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    image_rgb = image
                    
            # Use SigLIP processor for normalization and resizing
            inputs = self.processor(images=image_rgb, return_tensors="pt")
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
            return inputs
            
        except Exception as e:
            logging.error(f"Error preprocessing image for SigLIP: {e}")
            return None
            
    def encode_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Encode image to SigLIP embedding vector.
        
        Args:
            image: Input image as BGR numpy array
            
        Returns:
            Normalized embedding vector (768-dim for base model) or None if encoding fails
        """
        if not self.is_available():
            return None
            
        try:
            # Preprocess image
            inputs = self.preprocess_image(image)
            if inputs is None:
                return None

            # Forward pass through the model. Many SigLIP checkpoints define a
            # multimodal model that expects text inputs as well; calling
            # model(**inputs) may raise because 'input_ids' (text) is missing.
            # Attempt multiple strategies to obtain image embeddings without
            # requiring text input:
            # 1) model.get_image_features(**inputs) if available
            # 2) model.vision_model(**inputs) and extract pooler_output or CLS
            # 3) model(**inputs) and extract pooler_output or last_hidden_state
            image_embeds = None

            with torch.no_grad():
                # ── Strategy 1: get_image_features() ──────────────────────
                # Returns a tensor OR a ModelOutput object depending on the
                # transformers version — always extract the raw tensor.
                if hasattr(self.model, 'get_image_features'):
                    try:
                        raw = self.model.get_image_features(**inputs)
                        # If it's a ModelOutput (has .image_embeds or .pooler_output),
                        # extract the underlying tensor
                        if isinstance(raw, torch.Tensor):
                            image_embeds = raw
                        elif hasattr(raw, 'image_embeds') and raw.image_embeds is not None:
                            image_embeds = raw.image_embeds
                        elif hasattr(raw, 'pooler_output') and raw.pooler_output is not None:
                            image_embeds = raw.pooler_output
                        elif hasattr(raw, 'last_hidden_state'):
                            lh = raw.last_hidden_state
                            image_embeds = lh[:, 0, :] if lh.size(1) >= 1 else lh.mean(dim=1)
                        logging.debug('SigLIP: used get_image_features')
                    except Exception as e:
                        logging.debug(f"SigLIP get_image_features failed: {e}")

                # ── Strategy 2: vision_model submodule ─────────────────────
                if image_embeds is None and hasattr(self.model, 'vision_model'):
                    try:
                        vm     = self.model.vision_model
                        out_vm = vm(**{k: v for k, v in inputs.items()})
                        if hasattr(out_vm, 'pooler_output') and out_vm.pooler_output is not None:
                            image_embeds = out_vm.pooler_output
                        elif hasattr(out_vm, 'last_hidden_state'):
                            lh = out_vm.last_hidden_state
                            image_embeds = lh[:, 0, :] if lh.size(1) >= 1 else lh.mean(dim=1)
                        logging.debug('SigLIP: used vision_model submodule')
                    except Exception as e:
                        logging.debug(f"SigLIP vision_model forward failed: {e}")

                # ── Strategy 3: model(**inputs) fallback ───────────────────
                if image_embeds is None:
                    try:
                        outputs = self.model(**{k: v for k, v in inputs.items()})
                        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                            image_embeds = outputs.pooler_output
                        elif hasattr(outputs, 'last_hidden_state'):
                            lh = outputs.last_hidden_state
                            image_embeds = lh[:, 0, :] if lh.size(1) >= 1 else lh.mean(dim=1)
                        logging.debug('SigLIP: used model(**inputs) fallback')
                    except Exception as e:
                        logging.error(f"SigLIP model forward failed: {e}")
                        return None

                if image_embeds is None or not isinstance(image_embeds, torch.Tensor):
                    logging.error('SigLIP: could not extract a tensor from model output')
                    return None

                # ── Normalise for cosine similarity ────────────────────────
                norms        = torch.linalg.norm(image_embeds.float(), dim=-1, keepdim=True)
                image_embeds = image_embeds.float() / (norms + 1e-8)

                # Convert to numpy
                embedding = image_embeds.cpu().numpy().flatten()

            return embedding.astype(np.float32)
            
        except Exception as e:
            logging.error(f"Error encoding image with SigLIP: {e}")
            return None
            
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity score (-1 to 1, higher is more similar)
        """
        try:
            if emb1 is None or emb2 is None:
                return 0.0
                
            # Ensure embeddings are normalized
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
            
            # Compute cosine similarity
            similarity = float(np.dot(emb1_norm, emb2_norm))
            
            # Clamp to valid range
            return max(-1.0, min(1.0, similarity))
            
        except Exception as e:
            logging.error(f"Error computing SigLIP similarity: {e}")
            return 0.0
            
    def embed_crop(self, crop_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Convenience method to encode a cropped image region.
        
        Args:
            crop_image: Cropped image as BGR numpy array
            
        Returns:
            SigLIP embedding vector or None if encoding fails
        """
        return self.encode_image(crop_image)
        
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of SigLIP embeddings."""
        return self.embedding_dim
        
    def __str__(self) -> str:
        """String representation of the encoder."""
        status = "available" if self.is_available() else "unavailable"
        return f"SigLIPEncoder(model={self.model_name}, device={self.device}, status={status})"
        
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
    
def _resolve_local(model_ref: str) -> str:
        try:
            # If the provided value already points to an existing path, return it
            if os.path.exists(model_ref):
                return os.path.abspath(model_ref)
        except Exception:
            pass

        # If model_ref looks like a hub id (contains '/'), use the basename
        # as a candidate folder name (e.g. 'siglip-base-p16-224')
        candidate_name = os.path.basename(model_ref)

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        # Candidate at repo root
        cand1 = os.path.join(repo_root, candidate_name)
        if os.path.isdir(cand1):
            return os.path.abspath(cand1)

        # Search limited-depth for folders that include 'siglip' in their name.
        # Prefer folders whose name most closely matches the requested basename
        # (e.g. 'siglip-base-p16-224') over generic names like 'siglip_demo'.
        try:
            best_match = None
            best_score = -1
            for root, dirs, files in os.walk(repo_root):
                for d in dirs:
                    if 'siglip' not in d.lower():
                        continue
                    # Score by how many characters the folder name shares with candidate_name
                    score = sum(1 for c in candidate_name if c in d.lower())
                    # Exact match wins immediately
                    if d.lower() == candidate_name.lower():
                        return os.path.abspath(os.path.join(root, d))
                    if score > best_score:
                        best_score = score
                        best_match = os.path.abspath(os.path.join(root, d))
            if best_match:
                return best_match
        except Exception:
            pass

        # No local resolution found; return original ref (will cause error later)
        return model_ref      
    
def create_siglip_encoder(config: dict = None) -> SigLIPEncoder:
    """
    Factory function to create SigLIP encoder with configuration.
    
    Args:
        config: Configuration dictionary with optional keys:
                - siglip_model_name: Model name (default: google/siglip-base-patch16-224)
                - siglip_device: Device for inference (default: cpu)
                
    Returns:
        Initialized SigLIPEncoder instance
    """
    if config is None:
        config = {}
        
    model_name = config.get('siglip_model_name', 'google/siglip-base-patch16-224')
    device = config.get('siglip_device', 'cpu')

    resolved = _resolve_local(model_name)
    if resolved != model_name:
        logging.info(f"Resolved siglip_model_name '{model_name}' -> local path: {resolved}")
        model_name = resolved

    return SigLIPEncoder(model_name=model_name, device=device)


# Example usage and testing
if __name__ == "__main__":
    # Simple test of SigLIP encoder
    encoder = SigLIPEncoder()
    
    if encoder.is_available():
        print(f"SigLIP encoder loaded successfully: {encoder}")
        print(f"Embedding dimension: {encoder.get_embedding_dim()}")
        
        # Test with dummy image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        embedding = encoder.encode_image(test_image)
        
        if embedding is not None:
            print(f"Test embedding shape: {embedding.shape}")
            print(f"Test embedding norm: {np.linalg.norm(embedding):.4f}")
        else:
            print("Failed to encode test image")
    else:
        print("SigLIP encoder not available")