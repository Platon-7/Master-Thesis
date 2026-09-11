# Where the TikZ figure data came from

The MetaWorld and Robomimic curves in the workshop paper are **not** re-traced by
eye and **not** re-run. Both original figures were matplotlib vector PDFs, so the
exact plotted coordinates are stored inside them. These scripts read those
coordinates out and re-emit them as pgfplots.

    extract.py      decompress the PDF content stream; parse the graphics
                    operators into painted subpaths, tracking stroke/fill colour
                    (both RG and the gray G operator) and the q/Q state stack.
    splitpanels.py  map legend swatch colour -> model name, assign each subpath
                    to a panel, and convert device coords to data coords using
                    the tick-mark positions read from the same stream.
    gen.py          emit the pgfplots code, dropping RoboRef-ICL.

## Why the numbers are trustworthy

Three independent checks, all passed:

1. Extracted x values land exactly on the evaluation grid (5k..40k by 5k for
   MetaWorld), and y values on a clean 2-decimal grid -- an approximate transform
   would not do that.
2. The mean curve lies inside its own +-1 s.d. band at every point, though the
   two were extracted by different code paths (stroked path vs filled polygon).
3. The thesis text quotes "0.50 on Coffee-Push and 0.72 on Box-Close" for
   RoboRef-Asym. Those are the mean of the final three extracted evaluations
   (0.28, 0.60, 0.62 -> 0.500; 0.64, 0.65, 0.88 -> 0.723). The same holds for
   RoboRef-Std (0.11, 0.32) and Robometer-4B (0.00, 0.07), and for Robomimic's
   0.84 / 0.53 / 0.65 / 0.34.
