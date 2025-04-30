from collections import Counter

import torch
import sglang as sgl
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

def nllb_load(args):
    LANG2CODE = {
        "en": "eng_Latn",
        "de": "deu_Latn",
        "vi": "vie_Latn",
        "zh": "zho_Hans",
        "ja": "jpn_Jpan",
    }
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(
        args.translation_model_path, token=True, 
        src_lang=LANG2CODE[args.source_language], tgt_lang=LANG2CODE[args.target_language]
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.translation_model_path, device_map=device,
        torch_dtype=torch.float16, attn_implementation="flash_attention_2"
    )
    return device, tokenizer, model

def tower_load(args):
    device = "cuda"
    if args.beam_size == 1:
        model = sgl.Engine(
            model_path=args.translation_model_path,
            mem_fraction_static=0.4,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.translation_model_path, torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation="flash_attention_2"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.translation_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return device, tokenizer, model

def madlad_load(args):
    device = "cuda"
    tokenizer = T5Tokenizer.from_pretrained(args.translation_model_path)
    model = T5ForConditionalGeneration.from_pretrained(
        args.translation_model_path, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="flash_attention_2"
    )
    return device, tokenizer, model

def nllb_generate(self, src_prefix, tgt_prefix, return_all_beams=False):
    """
    Generate translation using NLLB.
    """
    inputs = self.tokenizer(
        src_prefix, 
        text_target=tgt_prefix, 
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    decoder_input_ids = torch.cat([inputs["labels"][:, -1:], inputs["labels"][:, :-1]], dim=-1)
    if inputs["input_ids"].size(0) > 1:
        decoder_input_ids = decoder_input_ids.repeat(inputs["input_ids"].size(0), 1)

    max_tokens = inputs["input_ids"].size(1) * self.max_len_a + self.max_len_b
    max_new_tokens = int(max_tokens - decoder_input_ids.size(1))

    if max_new_tokens <= 0:
        return [""] * inputs["input_ids"].size(0)

    tgt = self.model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        decoder_input_ids=decoder_input_ids,
        num_beams=self.beam_size,
        max_new_tokens=max_new_tokens,
        num_return_sequences=self.beam_size if return_all_beams else 1,
        no_repeat_ngram_size=3,
    )

    tgt = tgt[:, decoder_input_ids.size(1) : ]
    candidates = self.tokenizer.batch_decode(tgt, skip_special_tokens=True)
    candidates = [c.strip() for c in candidates]
    return candidates

def tower_generate(self, src_prefix, tgt_prefix, return_all_beams=False):
    LANG2ID = {
        "en": "English",
        "zh": "Chinese",
        "ja": "Japanese",
        "de": "German",
    }

    MT_TEMPLATE = """<|im_start|> user
Translate the following text from {} into {}.
{}: {}
{}:<|im_end|> 
<|im_start|> assistant
{}"""

    if type(src_prefix) is str:
        src_prefix = [src_prefix]

    src_lang_id = LANG2ID[self.source_language]
    tgt_lang_id = LANG2ID[self.target_language]

    prompts = [
        MT_TEMPLATE.format(
            src_lang_id, tgt_lang_id,
            src_lang_id, s,
            tgt_lang_id, tgt_prefix
        )
        for s in src_prefix
    ]

    inputs = self.tokenizer(
        prompts, 
        padding=True,
        return_tensors='pt'
    ).to('cuda')

    src_ids = self.tokenizer(src_prefix, padding=True, return_tensors='pt')
    tgt_ids = self.tokenizer(tgt_prefix, padding=True, return_tensors='pt')

    max_tokens = src_ids["input_ids"].size(1) * self.max_len_a + self.max_len_b
    max_new_tokens = int(max_tokens - tgt_ids["input_ids"].size(1))

    if max_new_tokens <= 0:
        return [""] * inputs["input_ids"].size(0)

    if self.beam_size == 1:
        sampling_params = {
            "max_new_tokens": max_new_tokens,
            "top_k": 1,
            "stop": ["\n"]
        }
        response = self.model.generate(
            prompts,
            sampling_params,
        )
        candidates = [o['text'] for o in response]
    else:
        tgt = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            num_beams=self.beam_size,
            num_return_sequences=self.beam_size if return_all_beams else 1
        )
        tgt = tgt[:, inputs["input_ids"].size(1) : ]
        candidates = self.tokenizer.batch_decode(tgt, skip_special_tokens=True)
    candidates = [c.strip() for c in candidates]

    return candidates

def madlad_generate(self, src_prefix, tgt_prefix, return_all_beams=False):
    if type(src_prefix) is str:
        src_prefix = [src_prefix]

    inputs = self.tokenizer(
        ["<2{}> ".format(self.target_language) + s for s in src_prefix], 
        text_target=tgt_prefix, 
        return_tensors="pt",
        padding=True,
    ).to(self.device)

    decoder_input_ids = torch.cat(
        [torch.tensor([[0]]).to(inputs["labels"]), inputs["labels"][:, :-1]], 
        dim=1
    )
    if inputs["input_ids"].size(0) > 1:
        decoder_input_ids = decoder_input_ids.repeat(inputs["input_ids"].size(0), 1)

    max_tokens = inputs["input_ids"].size(1) * self.max_len_a + self.max_len_b
    max_new_tokens = int(max_tokens - decoder_input_ids.size(1))

    if max_new_tokens <= 0:
        return [""] * inputs["input_ids"].size(0)

    tgt = self.model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        decoder_input_ids=decoder_input_ids,
        num_beams=self.beam_size,
        max_new_tokens=max_new_tokens,
        num_return_sequences=self.beam_size if return_all_beams else 1,
        no_repeat_ngram_size=3,
    )

    tgt = tgt[:, decoder_input_ids.size(1) : ]
    candidates = self.tokenizer.batch_decode(tgt, skip_special_tokens=True)
    candidates = [c.strip() for c in candidates]
    return candidates

def ralcp(candidates, threshold, tgt_char, is_token_ids=False):
    words = [c.split(' ') if not tgt_char else c for c in candidates] if not is_token_ids else candidates
    max_len = max(len(w) for w in words)
    translation = "" if not is_token_ids else torch.LongTensor([]).unsqueeze(0)
    for i in range(max_len):
        cnt = Counter(
            (' '.join(w[:i + 1]) if not tgt_char else w[:i + 1])
            if not is_token_ids else ' '.join(map(str, w[:i + 1].tolist()))
            for w in words
        )
        most_common_candidate = max(cnt, key=cnt.get)
        if cnt[most_common_candidate] >= len(candidates) * threshold:
            if is_token_ids:
                translation = torch.LongTensor(list(map(int, most_common_candidate.split(' ')))).unsqueeze(0)
            else:
                translation = most_common_candidate
        else:
            break
    return translation
    
def majority(candidates):
    cnt = Counter(candidates)
    most_common_candidate = max(cnt, key=cnt.get)
    return most_common_candidate, cnt[most_common_candidate]