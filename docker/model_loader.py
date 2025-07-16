import flair
import logging
import os
import torch

from flair.data import Sentence
from flair.models import SequenceTagger
from flair.nn import Model
from pathlib import Path
from somajo import SoMaJo
from timeit import default_timer as timer
from transformers import MT5ForConditionalGeneration, MT5TokenizerFast
from typing import List


os.environ["TOKENIZERS_PARALLELISM"] = "false"
flair.cache_root = Path(os.path.join(*['/app', 'flair_cache_root']))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CODEALLTAG_MT5_MODEL_DIR_LIST: List[str] = ["models", "codealltag", "mT5"]
CODEALLTAG_BILSTM_CRF_MODEL_DIR_LIST: List[str] = ["models", "codealltag", "BiLSTM_CRF"]
CODEALLTAG_GELECTRA_MODEL_DIR_LIST: List[str] = ["models", "codealltag", "GELECTRA"]
CODEALLTAG_SUPPORTED_TAGGERS: List[str] = ['codealltag_bilstmcrf', 'codealltag_gelectra']


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
        
        logger.info("Loading models...")
        try:
            self.codealltag_mT5_model = MT5ForConditionalGeneration.from_pretrained(os.path.join(*CODEALLTAG_MT5_MODEL_DIR_LIST))
            self.codealltag_mT5_tokenizer = MT5TokenizerFast.from_pretrained(os.path.join(*CODEALLTAG_MT5_MODEL_DIR_LIST))
            self.codealltag_mT5_model.eval()
            
            self.codealltag_bilstmcrf_tagger = SequenceTagger.load(os.path.join(*CODEALLTAG_BILSTM_CRF_MODEL_DIR_LIST, 'model.pt'))
            print(self.codealltag_bilstmcrf_tagger.embeddings)
            
            self.codealltag_gelectra_tagger = SequenceTagger.load(os.path.join(*CODEALLTAG_GELECTRA_MODEL_DIR_LIST, 'model.pt'))
            print(self.codealltag_gelectra_tagger.embeddings)
            
            logger.info("Models loaded successfully.")
        except Exception as e:
            logger.error("Failed to load models: {str(e)}")
            raise RuntimeError("Models loading failed!")
    
    def predict_with_codealltag_mT5(self, input_text):
        
        start = timer()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        device_stat = "CPU" if device == "cpu" else torch.cuda.get_device_name(0)
        print(f"device_stat: {device_stat}")
        
        tokenized_outputs = self.codealltag_mT5_tokenizer.batch_encode_plus(
            [input_text],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokenized_outputs["input_ids"]
        attention_mask = tokenized_outputs["attention_mask"]
        
        self.codealltag_mT5_model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outs = self.codealltag_mT5_model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   max_length=512,
                                   temperature=0.8,
                                   do_sample=True,
                                   top_k=100)
        dec = [
            self.codealltag_mT5_tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            for ids in outs
        ]
        
        end = timer()
        
        print(f"inference_time: {round(end - start, 3)}s")
        
        return dec[0]
    
    def predict_with_codealltag_tagger(self, tagger_id: str, sentences: List[Sentence]):
        
        if tagger_id == None or tagger_id not in CODEALLTAG_SUPPORTED_TAGGERS:
            return None
        
        if tagger_id == 'codealltag_bilstmcrf':
            self.codealltag_bilstmcrf_tagger.predict(sentences)
        elif tagger_id == 'codealltag_gelectra':
            self.codealltag_gelectra_tagger.predict(sentences)
        else:
            return None
        
        return sentences
        