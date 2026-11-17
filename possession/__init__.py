"""Architecture A: confidence-aware observation evidence + temporal state decoding.

Pipeline: raw ATD + Smart Tagging -> per-frame features -> soft H/A/D evidence -> per-half
Viterbi decoding -> frame,label CSV. numpy only; no training and no use of any ground truth.
See docs/ARCHITECTURE_A.md.
"""
