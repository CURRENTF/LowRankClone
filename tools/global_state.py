hyper_params = {}
ban_losses = []
ban_layers = []

data_cls_dict = {
    "general": 0,
    "med-en": 1,
    "med-zh+en": 2,
    "sft": 3,
    "pubmed_abs": 4,
    "meta_math": 5,
}

data_cls_reversed_dict = {
    v: k for k, v in data_cls_dict.items()
}

info_dict = {}
