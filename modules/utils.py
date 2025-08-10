import torch
from transformers import pipeline, Pipeline, AutoTokenizer, AutoModelForCausalLM


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Debug(metaclass=SingletonMeta):
    def __init__(self, debug=False):
        self.debug = debug

    def set_debug(self, value: bool):
        self.debug = value

    def print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)



class LLM(metaclass=SingletonMeta):
    def __init__(self, model_id: str = "google/gemma-3-1b-it"):
        self.model = None
        self.tokenizer = None
        self.model_id = model_id
        self.pipe = self.set_model_id(self.model_id)

    def set_model_id(self, model_id: str) -> Pipeline:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        return self.pipe

    def get_pipe(self):
        return self.pipe

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model
