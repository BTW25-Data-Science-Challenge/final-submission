from nbconvert import LatexExporter

notebook_path = 'report.ipynb'
# Exported from the mailing list, please change name appearance if wanted
authors = [
            'Hannes Bachmann',
            'Emmanuel Diehl',
            'Georg Gonsior',
            'Daniel Ricardo Gonzalez Villamizar',
            'Julius Hanusch',
            'Clara Hüfner',
            'Malte Maier-Knop',
            'Jordan Wenzel Richter',
            'Ansgar Seidemann',
            'Moritz Tschöpe',
            'Felix Wahler',
            ]

nl = '\\\\'
authorstring = ('\\author{'+nl+nl.join(authors)+
                nl * 3 + nl.join(['\\textit{Supervision by}', 'Jimmy Pöhlmann', 'Claudio Hartmann', 'Wolfgang Lehner']) + nl * 3 +'}')

try:
    latex_exporter = LatexExporter()
    body, resources = latex_exporter.from_filename(notebook_path)

    # remove Utils
    startutils = body.index('\\subsection{Utils}\\label{utils}') + len('\\subsection{Utils}\\label{utils}')
    endutils = body.index('\\subsection{Gathering Domain')
    utilsstring = '\\textit{code omitted - see notebook for the following contents:}\\begin{itemize}\\tightlist\n\\item installs and imports\n\\item base model structure\n\\item models implementation\n\\item benchmarking class\n\\item downloading data\n\\item loading data\n\\end{itemize}'
    body = body[:startutils] + utilsstring + body[endutils:]

    # pretty table formatting
    body = body.replace('\\begin{longtable}[]{@{}', "\\rowcolors{2}{white}{gray!25}\n\\begin{longtable}[]{@{}")
    body = body.replace('\\usepackage{graphicx}', '\\usepackage[table]{xcolor}\n\\usepackage{graphicx}')

    # fix title
    body = body.replace('\\title{report}','\\title{BTW 2025 Data Science Challenge}')
    body = body.replace('\\section{BTW 2025 Data Science\nChallenge}\\label{btw-2025-data-science-challenge}', '')

    # fix section hierarchy
    body = body.replace('\\subsection{', '\\section{')
    body = body.replace('\\subsubsection{', '\\subsection{')

    # fix table col spacing
    body = body.replace('>{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.1221}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.0560}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.0840}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.7379}}@{}}', \
                        '>{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.25}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.15}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.2}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.4}}@{}}')

    # fix CO₂
    body = body.replace('₂','$_{2}$')

    # fix figure in table col size
    body = body.replace('al{0.5769}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 2\\tabcolsep) * \\real{0.4231',
    'al{0.8}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 2\\tabcolsep) * \\real{0.2')
    #body = body.replace('\\pandocbounded{\\includegraphics[keepaspectratio]{src/attention_graphic.png}}',\
    #                    '\\raisebox{-0.8\\height}{\\includegraphics[keepaspectratio]{src/attention_graphic.png}}')

    # add authors
    body = body.replace('\\begin{document}', authorstring + '\n\\begin{document}')

    # euro sign fix
    body = body.replace('€', '\\euro{}')

    # add TOC and Abstract
    body = body.replace('\\section{Abstract}', '\\thispagestyle{empty}\n\\newpage\\begin{abstract}\n')
    body = body.replace('\\section{Introduction}','\\end{abstract}\n\\thispagestyle{empty}\n\\newpage\n\\thispagestyle{empty}\n\\hypersetup{linkcolor=black}\n\\tableofcontents\n\\thispagestyle{empty}\n\\newpage\\section{Introduction}')

    # fix pandoc overleaf issue
    body = body.replace('\\pandocbounded', '')
    body = body.replace('    \\makeatletter\n    \\newsavebox\\pandoc@box\n    \\newcommand*[1]{%\n      \\sbox\\pandoc@box{#1}%\n      % scaling factors for width and height\n      \\Gscale@div\\@tempa\\textheight{\\dimexpr\\ht\\pandoc@box+\\dp\\pandoc@box\\relax}%\n      \\Gscale@div\\@tempb\\linewidth{\\wd\\pandoc@box}%\n      % select the smaller of both\n      \\ifdim\\@tempb\\p@<\\@tempa\\p@\n        \\let\\@tempa\\@tempb\n      \\fi\n      % scaling accordingly (\\@tempa < 1)\n      \\ifdim\\@tempa\\p@<\\p@\n        \\scalebox{\\@tempa}{\\usebox\\pandoc@box}%\n      % scaling not needed, use as it is\n      \\else\n        \\usebox{\\pandoc@box}%\n      \\fi\n    }\n    \\makeatother', '')

    # fix unicode char
    body = body.replace('✅', '\\checkmark')

    # make section links work
    #body = body.replace('→ Data Analysis', '\\ref{data-analysis}~\\nameref{data-analysis}')
    #body = body.replace('→Data Cleaning/SMARD-Data Preprocessing', '\\ref{smard-data-preprocessing}~\\nameref{smard-data-preprocessing}')
    #body = body.replace('→SMARD\nElectricity Data', '\\ref{smard-electricity-market-data}~\\nameref{smard-electricity-market-data}')
    #body = body.replace('→SMARD Electricity\nData', '\\ref{smard-electricity-market-data}~\\nameref{smard-electricity-market-data}')
    #body = body.replace('→ Gathering\nDomain Knowledge', '\\ref{gathering-domain-knowledge}~\\nameref{gathering-domain-knowledge}')
    #body = body.replace('→ Gathering Domain Knowledge', '\\ref{gathering-domain-knowledge}~\\nameref{gathering-domain-knowledge}')
    #body = body.replace('→ Data Sources', '\\ref{data-sources}~\\nameref{data-sources}')
    #body = body.replace('→ Data\nSources', '\\ref{data-sources}~\\nameref{data-sources}')
    #body = body.replace('→ Visualization and Story Telling', '\\ref{visualization-and-story-telling}~\\nameref{visualization-and-story-telling}')
    #body = body.replace('→\nVisualization and Story Telling', '\\ref{visualization-and-story-telling}~\\nameref{visualization-and-story-telling}')
    #body = body.replace('→ Predictive Modelling', '\\ref{predictive-modeling}~\\nameref{predictive-modeling}')
    #body = body.replace('→ Summary and Future Work', '\\ref{summary-and-future-work}~\\nameref{summary-and-future-work}')
    #body = body.replace('→ Appendix I', '\\nameref{appendix-i-smard-dataset-columns}')
    #if (body.count('→') != 0): print("→ replacement error")

    # create appendix
    body = body.replace('\\section{Appendix I', '\\newpage\\appendix\n\\section{Appendix I')

    # fix table
    body = body.replace('  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.2645}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.3388}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.0331}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.3636}}',
                       '  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.4}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.2}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.2}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.2}}')

    # shorten TOC
    body = body.replace('\\subsection{Acknowledgement}','\\vspace{15em}\\subsection{Acknowledgement}')

    # set A4
    body = body.replace('\\documentclass[11pt]{article}','\\documentclass[a4paper]{article}')

    # scale down figure
    body = body.replace('{\\includegraphics[keepaspectratio]{src/compare_LSTM_models_on_val.png}}','{\\includegraphics[keepaspectratio, scale = 0.7]{src/compare_LSTM_models_on_val.png}}')

    # fix another link
    body = body.replace('https://www.smard.de/resource/blob/205652/63fcff2c9813096fa2229d769da164ef/smard-user-guide-09-2021-data.pdf',
                        '\\href{https://www.smard.de/resource/blob/205652/63fcff2c9813096fa2229d769da164ef/smard-user-guide-09-2021-data.pdf}{\\color{black}https://www.smard.de/resource/blob/205652/63fcff2c9813096fa2229d769da164ef/}\\hspace{0em}\\href{https://www.smard.de/resource/blob/205652/63fcff2c9813096fa2229d769da164ef/smard-user-guide-09-2021-data.pdf}{\\color{black}smard-user-guide-09-2021-data.pdf}')

    # smaller tables
    body = body.replace('\\begin{longtable}[]{@{}','{\\fontsize{8pt}{10pt}\\selectfont\\begin{longtable}[]{@{}')
    body = body.replace('\\end{longtable}','\\end{longtable}}')

    # smaller code blocks
    body = body.replace('\\begin{Verbatim}','\\begin{small}\n\\begin{Verbatim}')
    body = body.replace('\\end{Verbatim}','\\end{Verbatim}\n\\end{small}')

    # create links
    body = body.replace('{[}Netztransparenz, Index-Ausgleichspreis,\n2024{]}', '\\hyperref[bibliography]{[Netztransparenz, Index-Ausgleichspreis, 2024]}', 1)
    body = body.replace('{[}SMARD user guide, 2024{]}', '\\hyperref[bibliography]{[SMARD user guide, 2024]}', 1)
    body = body.replace('{[}European Commission, EU ETS, 2024{]}', '\\hyperref[bibliography]{[European Commission, EU ETS, 2024]}.', 1)
    body = body.replace('{[}Investing, Carbon Emissions Futures,\n2024{]}', '\\hyperref[bibliography]{[Investing, Carbon Emissions Futures, 2024]}', 1)
    body = body.replace('{[}Smard, Negative wholesale prices, 2025{]}', '\\hyperref[bibliography]{[Smard, Negative wholesale prices, 2025]}', 1)
    body = body.replace('{[}Finanztools,\nInflationsraten Deutschland, 2025{]}', '\\hyperref[bibliography]{[Finanztools, Inflationsraten Deutschland, 2025]}', 1)
    body = body.replace('{[}Smard, Großhandelspreise,\n2024{]}', '\\hyperref[bibliography]{[Smard, Großhandelspreise, 2024]}', 1)
    body = body.replace('{[}AutoGluon Forecasting Model Zoo, 2025{]}',
                        '\\hyperref[bibliography]{[AutoGluon Forecasting Model Zoo, 2025]}', 1)
    # check if the correct amount of links appear before the bibliography
    if (body[:body.index('label{bibliography}')].count('hyperref[bibliography') != 8): print("link creation error")

    # fix figure captions
    body = body.replace('\\DeclareCaptionFormat{nocaption}{}\n    \\captionsetup{format=nocaption,aboveskip=0pt,belowskip=0pt}','')

    # fix bib newpage
    body = body.replace('\\section{Bibliography}', '\\newpage\n\\section{Bibliography}')


    with open('output.tex', 'w', encoding='utf-8') as f:
        f.write(body)


    print("done")
except Exception as e:
    print(f"Error: {e}")
