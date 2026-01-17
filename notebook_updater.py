"""
Utility script for efficiently updating Jupyter notebooks.
Helps with adding cells and managing notebook content.
"""

import json
import sys
from typing import List, Dict, Any


def load_notebook(filepath: str) -> Dict[str, Any]:
    """Load a Jupyter notebook from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_notebook(filepath: str, notebook: Dict[str, Any]):
    """Save a Jupyter notebook to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)


def create_markdown_cell(content: str) -> Dict[str, Any]:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.split('\n') if '\n' in content else [content]
    }


def create_code_cell(content: str) -> Dict[str, Any]:
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content.split('\n') if '\n' in content else [content]
    }


def append_cells(notebook: Dict[str, Any], cells: List[Dict[str, Any]]):
    """Append cells to the end of a notebook."""
    notebook['cells'].extend(cells)


def insert_cells_at(notebook: Dict[str, Any], index: int, cells: List[Dict[str, Any]]):
    """Insert cells at a specific index."""
    for i, cell in enumerate(cells):
        notebook['cells'].insert(index + i, cell)


def get_cell_count(notebook: Dict[str, Any]) -> int:
    """Get the number of cells in a notebook."""
    return len(notebook['cells'])


def find_cell_by_content(notebook: Dict[str, Any], search_text: str) -> int:
    """Find the index of a cell containing specific text. Returns -1 if not found."""
    for i, cell in enumerate(notebook['cells']):
        source = cell.get('source', [])
        if isinstance(source, list):
            source = ''.join(source)
        if search_text in source:
            return i
    return -1


if __name__ == "__main__":
    print("Notebook Updater Utility")
    print("=" * 60)
    print("\nThis script provides utilities for managing Jupyter notebooks.")
    print("\nExample usage:")
    print("  from notebook_updater import load_notebook, save_notebook, create_markdown_cell")
    print("  nb = load_notebook('notebook.ipynb')")
    print("  # ... modify notebook ...")
    print("  save_notebook('notebook.ipynb', nb)")
