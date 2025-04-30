# TAF

This is the repository for the NAACL 2025 paper "Anticipating Future with Large Language Model for Simultaneous Machine Translation". 

## Environment

```bash
conda create -n taf-env -y python=3.12
conda activate taf-env

pip install uv
uv pip install "sglang[all]>=0.4.6.post1"
pip install pycryptodome accelerate protobuf simuleval # you might see compatibility issues with datasets and tqdm, ignore them
pip install flash-attn --no-build-isolation
pip install vllm

git clone git@github.com:ninja-build/ninja.git && cd ninja
git checkout release
./configure.py --bootstrap
```

## Inference

Example of Zh-En direction using Llama-3.1-8B-Instruct as prediction model and TowerInstruct-7B-v0.2 as translation model.

```bash
src_lang=zh
tgt_lang=en
src_file=data/zh_en/newstest2020-zhen.zh.sep
tgt_file=data/zh_en/newstest2020-zhen.en

if [ $tgt_lang != "zh" ]; then
    sacrebleu_tokenizer=13a
    eval_latency_unit=word
else
    sacrebleu_tokenizer=zh
    eval_latency_unit=char
fi

prediction_model_type=llama
prediction_model_path=meta-llama/Llama-3.1-8B-Instruct
prediction_num_continuations=10
prediction_max_tokens=10
prediction_top_k=-1
prediction_top_p=0.9

translation_model_type=tower
translation_model_path=Unbabel/TowerInstruct-7B-v0.2
beam_size=1
max_len_a=1.5
max_len_b=20

agree_thres=0.6
min_start=3

export PATH="$PWD/ninja:$PATH"
export PYTHONPATH="$PWD:$PYTHONPATH"

simuleval \
    --agent agents/ralcp_taf.py \
    --agent-class agents.RALCP_TAF \
    --source-language $src_lang \
    --target-language $tgt_lang \
    --source $src_file \
    --target $tgt_file \
    --sacrebleu-tokenizer $sacrebleu_tokenizer \
    --eval-latency-unit $eval_latency_unit \
    --prediction-model-type $prediction_model_type \
    --prediction-model-path $prediction_model_path \
    --prediction-num-continuations $prediction_num_continuations \
    --prediction-max-tokens $prediction_max_tokens \
    --prediction-top-k $prediction_top_k \
    --prediction-top-p $prediction_top_p \
    --translation-model-type $translation_model_type \
    --translation-model-path $translation_model_path \
    --beam-size $beam_size \
    --max-len-a $max_len_a \
    --max-len-b $max_len_b \
    --agree-thres $agree_thres \
    --min-start $min_start
```

## Citation

If you find this work useful, please cite it as follows:

```bibtex
@inproceedings{ouyang-etal-2025-anticipating,
    title = "Anticipating Future with Large Language Model for Simultaneous Machine Translation",
    author = "Ouyang, Siqi  and
      Hrinchuk, Oleksii  and
      Chen, Zhehuai  and
      Lavrukhin, Vitaly  and
      Balam, Jagadeesh  and
      Li, Lei  and
      Ginsburg, Boris",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.286/",
    pages = "5547--5557",
    ISBN = "979-8-89176-189-6",
    abstract = "Simultaneous machine translation (SMT) takes streaming input utterances and incrementally produces target text. Existing SMT methods only use the partial utterance that has already arrived at the input and the generated hypothesis. Motivated by human interpreters' technique to forecast future words before hearing them, we propose Translation by Anticipating Future (TAF), a method to improve translation quality while retaining low latency. Its core idea is to use a large language model (LLM) to predict future source words and opportunistically translate without introducing too much risk. We evaluate our TAF and multiple baselines of SMT on four language directions. Experiments show that TAF achieves the best translation quality-latency trade-off and outperforms the baselines by up to 5 BLEU points at the same latency (three words)."
}
```

## Contact

If you have any questions, please contact me at siqiouya@andrew.cmu.edu or raise GitHub issues.