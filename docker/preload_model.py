from model_loader import ModelLoader

print("Preloading the model...")
ModelLoader.get_instance()  # Load the model on startup
print("Model preloaded successfully!")