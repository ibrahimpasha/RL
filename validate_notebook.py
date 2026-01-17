import json

try:
    with open('reinforcement_learning_zero_to_hero.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"✓ Notebook is valid!")
    print(f"✓ Total cells: {len(nb['cells'])}")
    
    # Check for Double DQN content
    double_dqn_found = False
    for cell in nb['cells']:
        source = cell.get('source', [])
        if isinstance(source, list):
            source = ''.join(source)
        if 'Double DQN: Addressing Overestimation Bias' in source:
            double_dqn_found = True
            print("✓ Double DQN content found!")
            break
    
    if not double_dqn_found:
        print("✗ Double DQN content not found")
    
except Exception as e:
    print(f"✗ Error: {e}")
