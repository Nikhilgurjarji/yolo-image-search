from pathlib import Path
import json

def ensure_processed_dir(raw_path):
    raw_path = Path(raw_path)
    processed_path = raw_path.parent.parent/"processed" /raw_path.name
    processed_path.mkdir(parents=True,exist_ok=True)
    return processed_path

def save_metadata(metadata,raw_path):
    processed_path = ensure_processed_dir(raw_path)

    output_path = processed_path/ "metadata.json"
    with open(output_path ,'w') as f:
        json.dump(metadata,f)
    return output_path

def load_metadata(metadata_path):
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        processed_path = metadata_path.parent.parent/"processed" / metadata_path.name/"metadata.json"
        if processed_path.exists():
            metadata_path = processed_path
        else:
            raise FileNotFoundError(f"metadata not found at : ",{metadata_path})
    
    with open(metadata_path,'r') as f:
        return json.load(f)
    
def get_unique_class_counts(metadata):
    unique_classes = set()
    count_options = {}

    for item in metadata:
        for cls in item['detection']:
            unique_classes.add(cls['class'])
            if cls['class'] not in count_options:
                count_options[cls['class']] = set()
            count_options[cls['class']].add(cls['count'])
    
    unique_classes = sorted(unique_classes)
    for cls in count_options:
        count_options[cls] = sorted(count_options[cls])

    return unique_classes , count_options