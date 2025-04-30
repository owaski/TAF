from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction

import torch
import vllm

from agents.utils import (
    madlad_load,
    tower_load,
    nllb_load,
    madlad_generate, 
    tower_generate,
    nllb_generate,
)

class T2TBase(TextToTextAgent):
    """
    The agent generate the number of seconds from an input audio.
    """

    def __init__(self, args):
        super().__init__(args)
        self.source_segment_size = args.source_segment_size
        self.source_language = args.source_language
        self.target_language = args.target_language
        self.beam_size = args.beam_size
        self.max_len_a = args.max_len_a
        self.max_len_b = args.max_len_b

        self.src_char = self.source_language == 'zh' or self.source_language == 'ja'
        self.tgt_char = self.target_language == 'zh' or self.target_language == 'ja'

        self.translation_model_type = args.translation_model_type
        self.translation_model_path = args.translation_model_path
        self.load_translation_model(args)

        self.prediction_model_type = args.prediction_model_type
        self.prediction_model_path = args.prediction_model_path
        self.prediction_num_continuations = args.prediction_num_continuations
        self.prediction_max_tokens = args.prediction_max_tokens
        self.prediction_top_k = args.prediction_top_k
        self.prediction_top_p = args.prediction_top_p
        self.load_prediction_model(args)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--source-language", default="en", type=str)
        parser.add_argument("--target-language", default="es", type=str)

        # prediction model
        parser.add_argument("--prediction-model-type", type=str, default=None)
        parser.add_argument("--prediction-model-path", type=str, default=None)
        parser.add_argument("--prediction-num-continuations", type=int, default=10)
        parser.add_argument("--prediction-max-tokens", type=int, default=10)
        parser.add_argument("--prediction-top-k", type=int, default=10)
        parser.add_argument("--prediction-top-p", type=float, default=0.9)

        # translation model
        parser.add_argument("--translation-model-type", type=str, default=None)
        parser.add_argument("--translation-model-path", type=str, default=None)

        # generation
        parser.add_argument("--beam-size", type=int, default=4)
        parser.add_argument("--max-len-a", type=float, default=1.5)
        parser.add_argument("--max-len-b", type=float, default=20)

    def load_prediction_model(self, args):
        if args.prediction_model_type is None:
            return
        
        self.predict_model = vllm.LLM(
            model=args.prediction_model_path,
            gpu_memory_utilization=0.9,
            max_model_len=1024,
        )

        self.sampling_params = vllm.SamplingParams(
            max_tokens=self.prediction_max_tokens,
            top_k=self.prediction_top_k,
            top_p=self.prediction_top_p,
            n=self.prediction_num_continuations,
            stop=['\n']
        )

    def load_translation_model(self, args):
        func_map = {
            "madlad": madlad_load,
            "tower": tower_load,
            "nllb": nllb_load,
        }
        self.device, self.tokenizer, self.model = func_map[args.translation_model_type](args)

    def predict(self, src_prefix):
        response = self.predict_model.generate([src_prefix], self.sampling_params)
        batch_predictions = [src_prefix + o.text for o in response[0].outputs]
        return batch_predictions
    
    def generate(self, src_prefix, tgt_prefix, return_all_beams=False):
        func_map = {
            "madlad": madlad_generate,
            "tower": tower_generate,
            "nllb": nllb_generate,
        }
        return func_map[self.translation_model_type](self, src_prefix, tgt_prefix, return_all_beams)