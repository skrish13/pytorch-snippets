# Load pretrained weights
pretrained_dict = torch.load(pretrained_path)
# Get model state dicts
model_dict = model.state_dict()
# Filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# Overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# Load the new state dict
model.load_state_dict(model_dict)
