# TAF

This is the repository for the NAACL 2025 paper "Anticipating Future with Large Language Model for Simultaneous Machine Translation". 

## Environment

```bash
conda create -n taf -y python=3.10
conda activate taf

pip install vllm
pip install pycryptodome accelerate protobuf simuleval
pip install flash-attn --no-build-isolation

git clone git@github.com:ninja-build/ninja.git && cd ninja
git checkout release
./configure.py --bootstrap
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