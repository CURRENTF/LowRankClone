def assign_device_map(model_name: str, gpus=1, local_rank=None):
    model_name = model_name.lower()
    if gpus == 1 and local_rank is None:
        return "auto"
    elif gpus == 1:
        return f"cuda:{local_rank}"
    assert gpus == 2
    if "llama" in model_name and "8b" in model_name:
        return {
            "model.embed_tokens": 0,
            "model.zoom": 0,
            "model.layers.0": 0,
            "model.layers.1": 0,
            "model.layers.2": 0,
            "model.layers.3": 0,
            "model.layers.4": 0,
            "model.layers.5": 0,
            "model.layers.6": 0,
            "model.layers.7": 0,
            "model.layers.8": 0,
            "model.layers.9": 0,
            "model.layers.10": 0,
            "model.layers.11": 0,
            "model.layers.12": 0,
            "model.layers.13": 0,
            "model.layers.14": 0,
            "model.layers.15": 0,
            "model.layers.16": 0,
            "model.layers.17": 0,
            "model.layers.18": 0,
            "model.layers.19": 0,
            "model.layers.20": 1,
            "model.layers.21": 1,
            "model.layers.22": 1,
            "model.layers.23": 1,
            "model.layers.24": 1,
            "model.layers.25": 1,
            "model.layers.26": 1,
            "model.layers.27": 1,
            "model.layers.28": 1,
            "model.layers.29": 1,
            "model.layers.30": 1,
            "model.layers.31": 1,
            "model.norm": 1,
            "model.target_norm": 1,
            "lm_head": 1,
            "zoom_down": 1,
            "mseloss": 1,
        }
    else:
        raise ValueError
