# Copyright 2025 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import logging
from typing import List, Tuple

from sacremoses import MosesTokenizer, MosesDetokenizer
from simalign import SentenceAligner
import torch
import time


LOGGER = logging.getLogger('source_resegmenter.refiner')


def count_cross_alignments(
        split_str1: int, split_str2: int, alignments: List[Tuple[int, int]]) -> int:
    """
    Counts the number of cross-alignments based on a given index split.

    :param split_str1: Split index for the first string
    :param split_str2: Split index for the second string
    :param alignments: List of (idx1, idx2) pairs representing the mapping
    :return: Number of cross-alignments
    """
    cross_count = 0

    for idx1, idx2 in alignments:
        if (idx1 < split_str1 and idx2 >= split_str2) or \
                (idx1 >= split_str1 and idx2 < split_str2):
            cross_count += 1

    return cross_count


def find_optimal_source_split(
        alignments: List[Tuple[int, int]],
        n_source_words: int,
        target_split_idx: int) -> int:
    min_value = n_source_words + 1  # more than any possible value of cross-alignment
    argmin = -1
    # look for the src idx which minimizes the cross alignments:
    for s_idx in range(n_source_words - 1):
        result = count_cross_alignments(s_idx, target_split_idx, alignments)
        if result < min_value:
            min_value = result
            argmin = s_idx
    return argmin


def xlr_refine(source_texts: str, reference_texts: str, source_lang: str, target_lang: str) -> str:
    # tokenization before aligning source and target texts
    source_tokenizer = MosesTokenizer(source_lang)
    target_tokenizer = MosesTokenizer(target_lang)
    LOGGER.info(f"Tokenizing source texts in {source_lang} language")
    tokenized_source_texts = [
        source_tokenizer.tokenize(line, return_str=True) for line in source_texts.split("\n")]
    LOGGER.info(f"Tokenizing reference texts in {target_lang} language")
    tokenized_reference_texts = [
        target_tokenizer.tokenize(line, return_str=True) for line in reference_texts.split("\n")]

    # refinement of tokenized texts
    start = time.perf_counter()
    tokenized_refined_source_texts = _run_xlr_refine(
        tokenized_source_texts, tokenized_reference_texts)
    end = time.perf_counter()
    LOGGER.info(f"XLR refinement took {end - start:.6f} seconds")

    # de-tokenization after the refinement
    source_detokenizer = MosesDetokenizer(source_lang)
    detokenized_source_texts = "\n".join([
        source_detokenizer.detokenize(line.split(), return_str=True)
        for line in tokenized_refined_source_texts])
    return detokenized_source_texts


def _run_xlr_refine(source_texts: List[str], reference_texts: List[str]) -> List[str]:
    assert len(source_texts) == len(reference_texts), \
        "Expected the same number of lines in source and reference texts to re-align, instead " \
        f"we got {len(source_texts)} and {len(reference_texts)} lines, respectively."
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    LOGGER.info(f"Using {device} to compute sentence alignments")
    word_aligner = SentenceAligner(
        model="bert", token_type="bpe", matching_methods="mai", device=device)
    source_texts = source_texts.copy()  # avoid changing the content of the input list
    for i in range(len(source_texts) - 1):
        current_source_words = source_texts[i].strip().split()
        source_words = current_source_words + source_texts[i + 1].strip().split()
        current_target_words = reference_texts[i].strip().split()
        target_words = current_target_words + reference_texts[i + 1].strip().split()
        word_alignments = word_aligner.get_word_aligns(source_words, target_words)["itermax"]
        optimal_source_split = find_optimal_source_split(
            word_alignments, len(source_words), len(current_target_words))

        # if the optimal segmentation split is different from the previous one, update the current
        # and next source segment according to the new split point
        if len(current_source_words) != optimal_source_split:
            source_texts[i] = " ".join(source_words[:optimal_source_split])
            source_texts[i + 1] = " ".join(source_words[optimal_source_split:])
    return source_texts
