"""Assemble the supplied appendix plus the general rule and finished figures."""
from pathlib import Path
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "docs/appendix_notes/bandwidth_figures"
USER_SOURCE = Path('/Users/sam/.codex/attachments/ec79f374-507d-411f-9ec0-b0e6e1024ae1/Pasted text.txt')


def main():
    original = USER_SOURCE.read_text()
    insertion = (DEST/'bandwidth_rules_insert.tex').read_text()
    original = original.replace('% Override \\BandwidthFigurePath before input if the figure is stored elsewhere.\n', '')
    original = original.replace(r'\providecommand{\BandwidthFigurePath}{figures/bandwidth_appendix_figure.pdf}', '')
    # Replace the old diagnostic and its two descriptive references; keep the proof.
    original = original[:original.index(r'\begin{figure}[tbp]')]
    original = original.replace('Figure~\\ref{fig:bandwidth-appendix-diagnostic}(a) compares the corresponding\nbandwidth roots.',
        'The general-activation formula above specializes to this same first-pair\nscore; Figure~\\ref{fig:bw-width-rules} plots its bandwidth roots.')
    original = original.replace('Amplitude weighting matches the leading small-angle bound, which is',
        'For tanh, amplitude weighting matches the leading small-angle bound, which is')
    start = original.index('that regime. Figure~')
    original = original[:start]+'''that regime. Figure~\\ref{fig:bw-width-rules} displays the selected bandwidth
itself against $N$. Its slowly varying large-width curves should
not be read as a fixed-precision convergence theorem for the refined rule.
'''
    figures_at = insertion.index(r'\begin{figure}[tbp]')
    prose, figures = insertion[:figures_at], insertion[figures_at:]
    anchor = r'\subsubsection{Frequency-dependent pole bound}'
    assert original.count(anchor) == 1
    combined = original.replace(anchor, prose+'\n'+anchor)+'\n'+figures
    (DEST/'bandwidth_appendix_complete.tex').write_text(combined)

    # A standalone review copy spells out external representation-proof references.
    # The paper insertion retains all of those original labels and the citation key.
    preview = combined.replace(r'\subsection{Selection of the relative bandwidth}', r'\section{Selection of the relative bandwidth}')
    preview = preview.replace(r'\subsubsection', r'\subsection')
    preview = preview.replace('We derive the tanh bandwidth prescription from the pole contribution in\n\\qeqref{eq:corrected-decomp}.',
        "We derive the tanh bandwidth prescription from the pole contribution in\nthe representation proof's quadrature decomposition.")
    preview = preview.replace(r'\qeqref{eq:Pbound}', "the representation theorem's pole estimate")
    preview = preview.replace(r'\qeqref{eq:density} gives',
        r'the density $a_\gamma(z)=[f(z+id)-f(z-id)]/(2id)$ gives')
    preview = preview.replace('\\qeqref{eq:Rbounds}, as in Lemma~\\ref{lem:midpoint}\n\\cite[Secs.~4--5]{JavedTrefethen2014}',
        r'the midpoint quadrature factor bound $(e^{(2\ell+1)A_\lambda}-1)^{-1}$'+'\n'+
        r'(Javed and Trefethen, 2014, Sections 4--5)')
    preview = preview.replace(r'Section~\ref{sec:recovery}', 'the representation and numerical recovery analysis')
    labels = set(re.findall(r'\\label\{([^}]+)\}', preview))
    references = set(re.findall(r'\\(?:qeqref|eqref|ref)\{([^}]+)\}', preview))
    assert references-{'#1'} <= labels, references-labels-{'#1'}
    preamble = r'''\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,amssymb,amsthm,graphicx,booktabs,tabularx,microtype}
\usepackage[letterpaper,margin=0.8in]{geometry}
\usepackage[hidelinks]{hyperref}
\newtheorem{proposition}{Proposition}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{2em}
\allowdisplaybreaks
\begin{document}
'''
    (DEST/'bandwidth_appendix_preview.tex').write_text(preamble+preview+'\n\\end{document}\n')
    data_dir=ROOT/'results/checkpoint_C_geometry/expC09_bandwidth_figures/appendix_bandwidth/data'
    (DEST/'data').mkdir(exist_ok=True)
    for name in ['width_predictions.json','error_curves.json','refined_marker_fits_4x4.json','selected_refined_marker_fits.json','provenance.json','verification.json']:
        shutil.copy2(data_dir/name, DEST/'data'/name)
    print('Assembled insertion, complete subsection, standalone review source, and data.')


if __name__=='__main__': main()
