%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% amos.cls v0.2 06/22/2020
% Author: Andrey Pak apak33@gatech.edu
%			   Sam Fedeler samuel.fedeler@colorado.edu
% Alternate Contact: M. Holzinger marcus.holzinger@colorado.edu
%
% This is a preliminary version of AMOS conference LaTeX template.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{amos}[2020/06/22]

% Inherited from the standard article class
\LoadClass[10pt, letter]{article}

\usepackage{graphicx, titlesec, titling, etoolbox, mathptmx} % http://ctan.org/pkg/mathptmx
\usepackage[paper=letterpaper,margin=1in]{geometry}	% 1 inch margins

% TODO: Does not work at this point. Workaround in .tex document
\renewcommand{\maketitlehooka}{\bfseries\selectfont}

% Vertical spacing
\setlength{\headsep}{0pt}

% Authors and affiliations
\usepackage[blocks]{authblk}
\setlength{\affilsep}{0em}		% Interline spacing between the author and affiliation
\renewcommand*{\Affilfont}{\normalfont\itshape}		% Set affiliation to italic
\renewcommand*{\Authfont}{\bfseries}					 % Set author to bold

% Figure naming style
\renewcommand{\figurename}{Fig.}

% Abstract style
\renewcommand{\abstractname}{\normalsize{ABSTRACT}}
\renewenvironment{abstract}
{\vspace{-4em}\small
	\begin{center}
		\bfseries\abstractname\vspace{-.5em}\vspace{0.5ex}
	\end{center}
}

% Title style
\titleformat{\section}{\bfseries\center}{\thesection.}{1em}{\MakeUppercase}[]
\titleformat{\subsection}{\bfseries}{\thesubsection}{0.5em}{}[]
\titleformat{\subsubsection}{\bfseries}{\thesubsubsection}{0.5em}{}[]

% Section & subsection numbering
\renewcommand\thesection{\arabic{section}}
\renewcommand\thesubsection{\thesection.\arabic{subsection}}

% Paragraph intendation
\setlength{\parindent}{0pt}		   % No paragraph intendation
\setlength{\parskip}{0.5em}		  % Paragraph spacing
\setlength{\droptitle}{-0.5in}		% Title spacing

\pagenumbering{gobble}			% No page numbering

% Spacing between authors with different afiiliations
\makeatletter
\def\and{%
\end{tabular}%
\vspace{-1em}
\begin{tabular}[t]{c}}
\makeatother

% Other spacings
\titlespacing\section{0pt}{1.1\baselineskip}{\baselineskip}
\titlespacing\subsection{0pt}{0pt plus 4pt minus 2pt}{0pt plus 2pt minus 2pt}

% Temp Title Font Workaround
\newcommand*{\TitleFont}{%
	\usefont{\encodingdefault}{\rmdefault}{b}{n}%
	\fontsize{12}{20}%
	\selectfont}

%%%%%%%%%%%%%%%%%%%%
% BIBLIOGRAPHY
% Bibliography numbering
\makeatletter
\renewcommand\@biblabel[1]{#1.}
\makeatother
% Bibliography spacing
\let\OLDthebibliography\thebibliography
\renewcommand\thebibliography[1]{
	\OLDthebibliography{#1}
	\setlength{\parskip}{0pt}
	\setlength{\itemsep}{0pt}
}
\makeatletter
\renewcommand*{\@biblabel}[1]{\hfill[#1]}
\makeatother
%%%%%%%%%%%%%%%%%%%%
