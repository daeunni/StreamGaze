"""
Future Remind Hard Task Filtering

Filters:
1. Ambiguous objects (plastic, bag, container, etc.)
"""

from tqdm import tqdm


def filter_future_remind_hard(data, log_file=None):
    """Filter future remind hard tasks - remove ambiguous objects"""
    ambiguous_keywords = [
        'plastic', 'paper', 'cloth', 'fabric', 'material',
        'container', 'bag', 'box', 'case', 'package', 'wrapper',
        'thing', 'item', 'stuff', 'object', 'device', 'tool', 'equipment'
    ]
    
    stats = {
        'initial': len(data),
        'filtered_ambiguous_objects': 0,
        'final': 0
    }
    
    filtered_data = []
    
    for item in tqdm(data, desc="Filtering future_remind_hard"):
        q = item['questions'][0]
        target_obj = q.get('target_object', '').lower()
        
        # Check if ambiguous
        is_ambiguous = target_obj in ambiguous_keywords or \
                      any(target_obj == keyword + 's' for keyword in ambiguous_keywords)
        
        if is_ambiguous:
            stats['filtered_ambiguous_objects'] += 1
            if log_file:
                log_file.write(f"[FILTERED - AMBIGUOUS OBJECT] {q.get('target_object', '')}\n")
            continue
        
        filtered_data.append(item)
    
    stats['final'] = len(filtered_data)
    return filtered_data, stats

