import os
import tempfile
from pathlib import Path
from src.services.pdf_generator.generator import MarkdownPDFGenerator

def test_markdown_pdf_generation():
    generator = MarkdownPDFGenerator()
    query = "What is reinforcement learning?"
    answer_markdown = """
# Reinforcement Learning Synthesis

## Introduction
Reinforcement learning (RL) is a subfield of machine learning.

- **Agent**: The decision maker.
- **Environment**: The world the agent interacts with.

### Key Contributions
1. Policy gradient methods.
2. Q-learning.

## References
[1] Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction.
"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test_report.pdf"
        result_path = generator.generate_pdf(
            query=query,
            answer_markdown=answer_markdown,
            output_path=output_path
        )
        
        assert result_path.exists()
        assert result_path == output_path
        
        # Check PDF signature bytes (%PDF-1.)
        with open(result_path, "rb") as f:
            header = f.read(8)
            assert header.startswith(b"%PDF-1.")
