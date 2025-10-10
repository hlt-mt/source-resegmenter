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

import argparse
import logging

import mweralign

import source_resegmenter
from source_resegmenter import refiner

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGER = logging.getLogger('source_resegmenter.resegment')


def main(args: argparse.Namespace) -> None:
    with open(args.source_texts, 'r') as f:
        source_texts = f.read()
    with open(args.reference_texts, 'r') as f:
        reference_texts = f.read()
    with open(args.backtranslation_texts, 'r') as f:
        backtranslation_texts = f.read()
    assert len(backtranslation_texts.split("\n")) == len(reference_texts.split("\n")), \
        "Backtranslation texts must have the same number of lines as reference texts"

    # re-segment the source texts with the mwersegmenter
    logging.info(
        f"Resegmenting {args.source_texts} to match {args.backtranslation_texts} with mweralign")
    resegmented_source_texts = mweralign.align_texts(backtranslation_texts, source_texts)

    if args.segmeter == "xl-segmenter":
        with open(args.output, 'w') as f:
            f.write(resegmented_source_texts)
    elif args.segmeter == "xlr-segmenter":
        logging.info(f"Refining the segmentation with word alignments on {args.reference_texts}")
        refined_source_texts = refiner.xlr_refine(
            resegmented_source_texts, reference_texts, args.source_language, args.target_language)
        with open(args.output, 'w') as f:
            f.write(refined_source_texts)
    else:
        raise ValueError(f"Unknown segmenter: {args.segmeter}")


def cli_main():
    """
    Re-segmenter command-line interface (CLI) entry point.

    This function parses command-line arguments and starts the :func:`main` routine.

    Example usage::

        $ python resegment.py --source-texts asr_audio_1.en --reference-texts audio_1_ref.it \\
              --backtranslation-texts mt_audio_1_ref.en --output resegm_audio_1.en

    Command-line arguments:

    - ``--source-texts`` (str, required): Path to a txt file containing source texts.
    - ``--reference-texts`` (str, required): Path to a txt file containing the reference texts.
    - ``--backtranslation-texts`` (str, required): Path to a txt file containing the
        sentence-level translations of the reference texts.
    - ``--segmenter`` (str, optional): Type of segmenter to use. [Default: xlr-segmenter]
    - ``--output`` (str, required): Path to the output file.
    """
    LOGGER.info(f"Source resegmenter version: {source_resegmenter.__version__}")
    parser = argparse.ArgumentParser("source_resegmenter")
    parser.add_argument(
        "--source-texts", type=str, required=True,
        help="Path to a txt file containing source texts (one sentence per line) to be aligned to "
             "the reference texts.")
    parser.add_argument(
        "--reference-texts", type=str, required=True,
        help="Path to a txt file containing the reference texts (one sentence per line), to which "
             "the `--source-texts` have to be aligned.")
    parser.add_argument(
        "--backtranslation-texts", type=str, required=True,
        help="Path to a txt file containing the sentence-level translations into the source "
             "language of the reference texts.")
    parser.add_argument(
        "--segmenter", choices=["xl-segmenter", "xlr-segmenter"], default="xlr-segmenter",
        help="Type of segmenter to use. `xl-segmenter` refers to the mwersegmenter between the "
             "source texts and the backtranslation texts, while `xlr-segmenter` refines this "
             "segmentation by means of word-level alignments between the intermediate result and "
             "the reference texts. [Default: xlr-segmenter]")
    parser.add_argument(
        "--source-language", type=str, required=False, default="en",
        help="Language of the source texts in two-digit code.")
    parser.add_argument(
        "--target-language", type=str, required=False, default="en",
        help="Language of the target texts in two-digit code.")
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output file.")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
