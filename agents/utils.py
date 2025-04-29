import torch

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
    return candidates