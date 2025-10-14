Usage
=====

This library assumes that 3 txt files are available:

1. *The source text to be re-segmented*, whose segmentation into lines has to be refined to match
   that of a reference file;
2. *The reference text*, to which we want to obtain a line-level alignment of the source text;
3. *A backtranslation* of the reference text into the source language, aligned at the line level
   with the reference text.

Once these three txt files are available, this tool can be used from command line as::

    source_resegmenter --source-texts asr_audio_1.en --reference-texts audio_1_ref.it \
        --backtranslation-texts mt_audio_1_ref.en --output resegm_audio_1.en
