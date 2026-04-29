# IEEE Report

This folder contains the IEEE-style project report for the Tox21 MLOps pipeline.

Files:

- `ieee_tox21_mlops_report.tex`: IEEEtran LaTeX source.
- `ieee_tox21_mlops_report.pdf`: compiled report.

Compile locally from this folder:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error ieee_tox21_mlops_report.tex
pdflatex -interaction=nonstopmode -halt-on-error ieee_tox21_mlops_report.tex
```

If `pdflatex` is not available in a new terminal session, use the installed MiKTeX binary directly:

```powershell
& "C:\Users\Home\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode -halt-on-error ieee_tox21_mlops_report.tex
```

Before final academic submission, fill the `TODO` author, institution, course, instructor, email, and submission-date metadata in the LaTeX source.
