from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction

import torch

from agents.ralcp import RALCP
from agents.utils import ralcp

@entrypoint
class RALCP_TAF(RALCP):
    def __init__(self, args):
        super().__init__(args)
        
    @torch.inference_mode()
    def policy(self, states: Optional[AgentStates] = None):
        if states is None:
            states = self.states

        if len(states.source) < self.min_start and not states.source_finished:
            return ReadAction()

        src_prefix = " ".join(states.source) if self.source_language != 'zh' else ''.join(states.source)
        tgt_prefix = " ".join(states.target) if self.target_language != 'zh' else ''.join(states.target)

        batch_predictions = []
        candidates = []
        if states.source_finished:         
            translation = self.generate(src_prefix, tgt_prefix)[0]
        else:
            batch_predictions = self.predict(src_prefix)
            candidates = self.generate(batch_predictions, tgt_prefix, return_all_beams=True)
            translation = ralcp(candidates, self.agree_thres, self.tgt_char)

        print(src_prefix, tgt_prefix + ' ' + translation, sep='\n')

        if translation != "" or states.source_finished:
            return WriteAction(
                content=translation,
                finished=states.source_finished,
            )
        else:
            return ReadAction()