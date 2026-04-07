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
from abc import abstractmethod
from typing import List, Tuple

import torch
import time


LOGGER = logging.getLogger('source_resegmenter.refiner')


class Refiner(object):
    @abstractmethod
    def find_optimal_split(
            self,
            source_words: List[str],
            target_words: List[str],
            target_split_idx) -> int:
        """

        :param source_words: list of words in the current and next line of the candidate
            transcripts
        :param target_words: list of words in the current and next line of the gold references
        :param target_split_idx: index of next split among current and next line in the gold
            reference
        :return: the refined index of the best split among current and next line in the candidate
            transcripts
        """
        ...


class SimAlignRefiner(Refiner):
    """
    Refiner class leveraging `SimAlign <https://aclanthology.org/2020.findings-emnlp.147/>`_ word
    alignments to find the best split in candidate transcripts.
    """
    def __init__(self):
        from simalign import SentenceAligner

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        LOGGER.info(f"Using {device} to compute sentence alignments")
        self.word_aligner = SentenceAligner(
            model="bert", token_type="bpe", matching_methods="mai", device=device)

    def find_optimal_split(
            self,
            source_words: List[str],
            target_words: List[str],
            target_split_idx) -> int:
        word_alignments = self.word_aligner.get_word_aligns(source_words, target_words)["itermax"]
        n_source_words = len(source_words)
        min_value = n_source_words + 1  # more than any possible value of cross-alignment
        argmin = -1
        # look for the src idx which minimizes the cross alignments:
        for s_idx in range(n_source_words - 1):
            result = SimAlignRefiner.count_cross_alignments(
                s_idx, target_split_idx, word_alignments)
            if result < min_value:
                min_value = result
                argmin = s_idx
        return argmin

    @staticmethod
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


class LaBSERefiner(Refiner):
    """
    Refiner class leveraging `LaBSE <https://aclanthology.org/2022.acl-long.62/>`_ word embeddings
    to find the best split in candidate transcripts.
    """
    def __init__(self, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        LOGGER.info(f"Using {device} to compute LaBSE word embeddings")
        self.batch_size = batch_size
        self.labse = SentenceTransformer('sentence-transformers/LaBSE', device=device)

    def find_optimal_split(
            self,
            source_words: List[str],
            target_words: List[str],
            target_split_idx) -> int:
        import numpy as np

        target_current = " ".join(target_words[:target_split_idx])
        target_next = " ".join(target_words[target_split_idx:])
        emb_target = self.labse.encode([target_current, target_next])
        emb_target = [e / np.linalg.norm(e) for e in emb_target]
        # build source sentences candidates
        src_candidates = []
        for s_idx in range(len(source_words) - 1):
            src_candidates.append(
                (" ".join(source_words[:s_idx]), " ".join(source_words[s_idx:])))

        # compute all possible embeddings
        emb_current = []
        emb_next = []
        for batch_start_idx in range(0, len(source_words), self.batch_size):
            batch_src_cand = src_candidates[batch_start_idx:batch_start_idx + self.batch_size]
            batch_emb_current = self.labse.encode([s[0] for s in batch_src_cand])
            batch_emb_next = self.labse.encode([s[1] for s in batch_src_cand])
            emb_current.extend([e / np.linalg.norm(e) for e in batch_emb_current])
            emb_next.extend([e / np.linalg.norm(e) for e in batch_emb_next])

        assert len(emb_current) == len(emb_next) == len(source_words) - 1

        # find splitting point that minimizes cosine distance
        min_value = 2  # more than any possible value of cosine distance
        argmin = -1
        for s_idx in range(len(source_words) - 1):
            cosine_distance_current = 1 - np.dot(emb_current[s_idx], emb_target[0])
            cosine_distance_next = 1 - np.dot(emb_next[s_idx], emb_target[1])
            result = (cosine_distance_current + cosine_distance_next) / 2

            if result < min_value:
                min_value = result
                argmin = s_idx
        return argmin


def xlr_simalign(
        source_texts: str, reference_texts: str, source_lang: str, target_lang: str) -> str:
    # tokenization before aligning source and target texts
    from sacremoses import MosesTokenizer, MosesDetokenizer

    source_tokenizer = MosesTokenizer(source_lang)
    target_tokenizer = MosesTokenizer(target_lang)
    LOGGER.info(f"Tokenizing source texts in {source_lang} language")
    tokenized_source_texts = [
        source_tokenizer.tokenize(line, return_str=True) for line in source_texts.split("\n")]
    LOGGER.info(f"Tokenizing reference texts in {target_lang} language")
    tokenized_reference_texts = [
        target_tokenizer.tokenize(line, return_str=True) for line in reference_texts.split("\n")]

    # refinement of tokenized texts
    tokenized_refined_source_texts = _run_xlr_refine(
        SimAlignRefiner(), tokenized_source_texts, tokenized_reference_texts)

    # de-tokenization after the refinement
    source_detokenizer = MosesDetokenizer(source_lang)
    detokenized_source_texts = "\n".join([
        source_detokenizer.detokenize(line.split(), return_str=True)
        for line in tokenized_refined_source_texts])
    return detokenized_source_texts


def xlr_labse(source_texts: str, reference_texts: str) -> str:
    tokenized_source_texts = [
        line for line in source_texts.split("\n")]
    tokenized_reference_texts = [
        line for line in reference_texts.split("\n")]

    # refinement of tokenized texts
    tokenized_refined_source_texts = _run_xlr_refine(
        LaBSERefiner(), tokenized_source_texts, tokenized_reference_texts)

    # de-tokenization after the refinement
    detokenized_source_texts = "\n".join([
        line
        for line in tokenized_refined_source_texts])
    return detokenized_source_texts


def _run_xlr_refine(
        refiner: Refiner, source_texts: List[str], reference_texts: List[str]) -> List[str]:
    assert len(source_texts) == len(reference_texts), \
        "Expected the same number of lines in source and reference texts to re-align, instead " \
        f"we got {len(source_texts)} and {len(reference_texts)} lines, respectively."
    start = time.perf_counter()
    source_texts = source_texts.copy()  # avoid changing the content of the input list
    for i in range(len(source_texts) - 1):
        current_source_words = source_texts[i].strip().split()
        source_words = current_source_words + source_texts[i + 1].strip().split()
        current_target_words = reference_texts[i].strip().split()
        target_words = current_target_words + reference_texts[i + 1].strip().split()
        optimal_source_split = refiner.find_optimal_split(
            source_words, target_words, len(current_target_words))

        # if the optimal segmentation split is different from the previous one, update the current
        # and next source segment according to the new split point
        if len(current_source_words) != optimal_source_split:
            source_texts[i] = " ".join(source_words[:optimal_source_split])
            source_texts[i + 1] = " ".join(source_words[optimal_source_split:])
        source_texts[i] = source_texts[i].strip()
        source_texts[i + 1] = source_texts[i + 1].strip()
    end = time.perf_counter()
    LOGGER.info(f"Refinement took {end - start:.6f} seconds")
    return source_texts
