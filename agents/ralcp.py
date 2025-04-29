from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction

import torch

from agents.base import T2TBase
from agents.utils import ralcp

@entrypoint
class RALCP(T2TBase):
    def __init__(self, args):
        super().__init__(args)
        self.agree_thres = args.agree_thres
        self.min_start = args.min_start

    @staticmethod
    def add_args(parser):
        T2TBase.add_args(parser)
        parser.add_argument("--agree-thres", type=float, default=0.5)
        parser.add_argument("--min-start", type=int, default=1)
        
    @torch.inference_mode()
    def policy(self, states: Optional[AgentStates] = None):
        if states is None:
            states = self.states

        if len(states.source) < self.min_start and not states.source_finished:
            return ReadAction()

        src_prefix = " ".join(states.source) if self.source_language != 'zh' else ''.join(states.source)
        tgt_prefix = " ".join(states.target) if self.target_language != 'zh' else ''.join(states.target)

        if states.source_finished:  
            translation = self.generate(src_prefix, tgt_prefix)[0]
        else:
            candidates = self.generate(src_prefix, tgt_prefix, return_all_beams=True)
            translation = ralcp(candidates, self.agree_thres, self.tgt_char)

        print(src_prefix, tgt_prefix + ' ' + translation, sep='\n')

        if translation != "" or states.source_finished:
            return WriteAction(
                content=translation,
                finished=states.source_finished,
            )
        else:
            return ReadAction()