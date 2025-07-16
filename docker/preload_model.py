from model_loader import ModelLoader

print("Preloading the models...")
ModelLoader.get_instance()  # Load models on startup
print("Models preloaded successfully!")