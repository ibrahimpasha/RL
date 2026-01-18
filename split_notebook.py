import json

# Read the notebook
with open('reinforcement_learning_zero_to_hero.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the split point (Section 3)
split_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if 'Section 3: Advanced Topics' in source and '<a id=' in source:
            split_idx = i
            break

if split_idx is None:
    print("Could not find Section 3 marker")
    exit(1)

print(f'Total cells: {len(nb["cells"])}')
print(f'Split at cell: {split_idx}')
print(f'Part 1 cells: {split_idx}')
print(f'Part 2 cells: {len(nb["cells"]) - split_idx}')

# Create Part 1 (Sections 1-2)
nb_part1 = {
    'cells': nb['cells'][:split_idx],
    'metadata': nb['metadata'],
    'nbformat': nb['nbformat'],
    'nbformat_minor': nb['nbformat_minor']
}

# Add conclusion cell to Part 1
conclusion_cell = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## End of Part 1\n',
        '\n',
        'This concludes Part 1 of the Reinforcement Learning: Zero to Hero notebook.\n',
        '\n',
        '**What you learned:**\n',
        '- Section 1: Foundational Concepts (Bandits, MDPs, Dynamic Programming)\n',
        '- Section 2: Core Algorithms (Monte Carlo, TD Learning, Q-Learning, DQN)\n',
        '\n',
        '**Continue to Part 2 for:**\n',
        '- Section 3: Advanced Topics\n',
        '- Section 4: Code Implementations\n',
        '- Section 5: Real-World Applications\n',
        '- Section 6: Advanced Research & Deployment\n'
    ]
}
nb_part1['cells'].append(conclusion_cell)

# Create Part 2 (Sections 3-6)
intro_cell = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '# Reinforcement Learning: Zero to Hero - Part 2\n',
        '\n',
        '## Advanced Topics, Applications, and Research\n',
        '\n',
        'This is Part 2 of the Reinforcement Learning: Zero to Hero notebook.\n',
        '\n',
        '**Prerequisites:** Complete Part 1 first, which covers:\n',
        '- Section 1: Foundational Concepts\n',
        '- Section 2: Core Algorithms\n',
        '\n',
        '**This notebook covers:**\n',
        '- Section 3: Advanced Topics\n',
        '- Section 4: Code Implementations\n',
        '- Section 5: Real-World Applications\n',
        '- Section 6: Advanced Research & Deployment\n',
        '\n',
        '---\n'
    ]
}

nb_part2 = {
    'cells': [intro_cell] + nb['cells'][split_idx:],
    'metadata': nb['metadata'],
    'nbformat': nb['nbformat'],
    'nbformat_minor': nb['nbformat_minor']
}

# Save Part 1
with open('reinforcement_learning_part1_foundations.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_part1, f, indent=1, ensure_ascii=False)

# Save Part 2
with open('reinforcement_learning_part2_advanced.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_part2, f, indent=1, ensure_ascii=False)

print('\nNotebooks created successfully!')
print('- reinforcement_learning_part1_foundations.ipynb')
print('- reinforcement_learning_part2_advanced.ipynb')
