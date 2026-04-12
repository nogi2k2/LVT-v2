"""Lightweight package initializer for label_verifier.

Avoid importing heavy submodules at package import time (for example
pdf_to_image which requires PyMuPDF). Submodules should be imported by
consumers (e.g., GUI or scripts) via explicit imports when needed.
"""

# Expose a minimal public surface. Consumers should import submodules
# explicitly (for example: `from label_verifier import controller`).
__all__ = []
