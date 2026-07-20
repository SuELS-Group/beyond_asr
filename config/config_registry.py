import datetime
import os
import yaml

MODELS = {
    "qwen0.5b": "Qwen/Qwen1.5-0.5B",
    "qwen0.6b": "Qwen/Qwen3-0.6B",
    "qwen2b": "Qwen/Qwen3.5-2B-Base",
    "qwen9b": "Qwen/Qwen3.5-9B-Base",
    "qwen14b": "Qwen/Qwen2.5-14B",
    "llama1b": "meta-llama/Llama-3.2-1B",
    "llama3b": "meta-llama/Llama-3.2-3B",
    "llama13b": "meta-llama/Llama-2-13b-hf"
}

ID_DATASETS = {
    "standard": "keivalya/MedQuad-MedicalQnADataset",
}

ATTN_ONLY = ["q_proj", "k_proj", "v_proj", "o_proj"]

MLP_ONLY = ["gate_proj", "up_proj", "down_proj"]

TARGET_MODULES = {
    "qwen": ATTN_ONLY + MLP_ONLY,
    "llama": ATTN_ONLY + MLP_ONLY 
}

ALLOWED_R = [2, 4, 8, 16, 32]

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def get_target_modules(model_name: str) -> list:
    """Get target module by model family"""
    if model_name.startswith("qwen"):
        return TARGET_MODULES["qwen"]
    if model_name.startswith("llama"):
        return TARGET_MODULES["llama"]
    raise ValueError(f"Unknow model or family not supported : {model_name}")


# =============================================================================
# 3. VALIDATE CONFIG
# =============================================================================

def validate_config(config: dict) -> bool:
    """Strict validation of config dict"""
    
    required_keys = ["model_name", "dataset_name", "trainer_name", "r", "lora_alpha"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"❌ Missing parameter : {key}")

    if config["model_name"] not in MODELS:
        raise ValueError(f"❌ model_name invalid. Choices : {list(MODELS.keys())}")
        
    if config["dataset_name"] not in ID_DATASETS:
        raise ValueError(f"❌ dataset_name invalid. Choices : {list(ID_DATASETS.keys())}")

    if config["r"] not in ALLOWED_R:
        print(f"⚠️ Attention: r={config['r']} isn't in standard values {ALLOWED_R}")

    return True


# =============================================================================
# 4. BUILD CONFIG
# =============================================================================

def build_config(
    model_name: str, 
    r: int, 
    lora_alpha: int, 
    lora_dropout: float = 0.05,
    dataset_name: str = "standard", 
    trainer_name: str = "lora", 
    batch_size: int = 16,
    lr: float = 5e-6,
    epochs: int = 15,
    max_length: int = 512,
    seed: int = 2026,
) -> dict:
    """
    Get raw parameters, validate then generate final dict ready to be injected in train scripts
    """
    
    raw_params = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "trainer_name": trainer_name,
        "r": r,
        "lora_alpha": r*2,
    }
    
    validate_config(raw_params)

    config = {
        "seed": seed,
        "trainer_name": trainer_name,
        "model_name_or_path": MODELS[model_name],
        "dataset_name": ID_DATASETS[dataset_name],
        "max_length": max_length,
    }

    folder_name = f"outputs/saved_models/{model_name}_{dataset_name}_{trainer_name}_r{r}_alpha{lora_alpha}"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config["output_dir"] = f"{folder_name}_{timestamp}"

    config.update({"lora":{
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": get_target_modules(model_name)
    }})
    
    config.update({"train_arguments": {
        "num_train_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr
    }})

    return config

# =============================================================================
# 3. SAVE CONFIG
# =============================================================================

def save_config(config: dict, output_dir: str):
    """Save used configuration"""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "run_config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"📄 Configuration saved in : {config_path}")