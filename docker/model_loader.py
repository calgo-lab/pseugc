import logging
import os
import torch
from timeit import default_timer as timer
from transformers import MT5ForConditionalGeneration, MT5TokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CHECKPOINT_DIR = "model_checkpoint_dir"

class ModelLoader:
    
    _instance = None
    
    @staticmethod
    def get_instance():
        if ModelLoader._instance is None:
            ModelLoader._instance = ModelLoader()
        return ModelLoader._instance
    
    def __init__(self):
        if ModelLoader._instance is not None:
            raise Exception("This class is a singleton! Use get_instance() instead.")
        
        logger.info("Loading PyTorch Model...")
        try:
            self.model = MT5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT_DIR)
            self.tokenizer = MT5TokenizerFast.from_pretrained(MODEL_CHECKPOINT_DIR)
            
            self.model.eval()
            
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load model: {str(e)}")
            raise RuntimeError("Model loading failed!")
    
    def predict(self, input_text):
        
        start = timer()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        device_stat = "CPU" if device == "cpu" else torch.cuda.get_device_name(0)
        print(f"device_stat: {device_stat}")
        
        tokenized_outputs = self.tokenizer.batch_encode_plus(
            [input_text],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokenized_outputs["input_ids"]
        attention_mask = tokenized_outputs["attention_mask"]
        
        self.model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outs = self.model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   max_length=512,
                                   temperature=0.8,
                                   do_sample=True,
                                   top_k=100)
        dec = [
            self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            for ids in outs
        ]
        
        end = timer()
        
        print(f"inference_time: {round(end - start, 3)}s")
        
        return dec[0]