"""Side-by-side display of the exact submitted figure and revised comparison.

Both images are shown at equal height without stretching, to compare styling.
The reference PNG is extracted directly from page 4 of submission (18).
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_C_geometry/expC14_theory_solver_comparison/five_function_error_norm_panel'


def main():
    paths=[OUT/'final_paper_reference.png',OUT/'comparison.png']
    images=[Image.open(p).convert('RGB') for p in paths]
    aspects=[im.width/im.height for im in images]
    fig=plt.figure(figsize=(sum(aspects)*3.2+.15,3.55))
    total=sum(aspects)*3.2+.15
    for index,(im,aspect,title) in enumerate(zip(images,aspects,
            ['Final figure in submitted paper','Revised comparison'])):
        left=0 if index==0 else (aspects[0]*3.2+.15)/total
        ax=fig.add_axes([left,.01,aspect*3.2/total,3.2/3.55])
        ax.imshow(im);ax.set_axis_off();ax.set_title(title,fontsize=12,fontweight='bold',pad=7)
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'style_comparison.{ext}',dpi=240,facecolor='white')
    plt.close(fig)


if __name__=='__main__':main()
