# Source Resegmenter

``source_resegmenter`` is a Python library for re-segmenting a text into lines in a way
that it matches a reference text in another language.

The repository is tested using Python 3.11. Although it may work also with other Python versions,
we do not ensure compatibility with them. Check out the [Usage](#Usage) section for instructions on
how to use the repository and the [Installation](#Installation) section for further information
about how to install the project.


## Installation

You can install the latest stable version from PyPI:

```shell
pip install source_resegmenter
```

Or, to install from source:

```shell
git clone https://github.com/hlt-mt/source_resegmenter.git
cd source_resegmenter
pip install .
```

For development (with docs and testing tools):

```shell
pip install -e .[dev]
```

## Usage

This library assumes that 3 txt files are available:

1. _The source text to be re-segmented_, whose segmentation into lines has to be refined to match
   that of a reference file;
2. _The reference text_, to which we want to obtain a line-level alignment of the source text;
3. _A backtranslation_ of the reference text into the source language, aligned at the line level
   with the reference text.

Once these three txt files are available, this tool can be used from command line as:

```shell
source_resegmenter --source-texts asr_audio_1.en --reference-texts audio_1_ref.it \
    --backtranslation-texts mt_audio_1_ref.en --output resegm_audio_1.en
```

## Contributing

Contributions from interested researchers and developers are extremely appreciated.

You can create an ***issue*** in case of problems with the code, questions, or feature requests.
You are also more than welcome to create a ***pull request*** that addresses any ***issue***.

## Licence

``source_resegmenter_`` is licensed under [Apache Version 2.0](LICENSE). 

## Credits
If you use this library, please cite:


```
@inproceedings{cettolo-et-al-2025-xlr-segmenter,
    title={{On the Reliability of Source-Aware Metrics with Synthetic Transcripts in Speech Translation Evaluation}},
    author={Cettolo, Mauro and Gaido, Marco and Negri, Matteo and Papi, Sara and Bentivogli, Luisa},
    booktitle = "",
    address = "",
    year={2025}
}
```

