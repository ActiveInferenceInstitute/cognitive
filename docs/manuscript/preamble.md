# LaTeX preamble

The builder extracts the fenced LaTeX block and passes it to Pandoc and
XeLaTeX. The packages are limited to functionality used by the source.

```latex
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{booktabs}
\usepackage{float}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{hyperref}
\hypersetup{colorlinks=true,allcolors=blue}
\usepackage[capitalise,noabbrev]{cleveref}
```
