import yaml

def load_config(config_path = "configs/default.yaml"):
    with open(config_path ,'r') as f:
        configs = yaml.safe_load(f)
        return configs
    
def save_config(configs ,config_path = "configs/default.yaml"):
    with open(config_path ,'w') as f:
        yaml.dump(configs,f)
        return configs