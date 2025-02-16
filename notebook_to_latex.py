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
    body = body[:startutils] + '\\textit{code omitted - see notebook}' + body[endutils:]

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

    # add TOC
    body = body.replace('\\maketitle', '\\maketitle\n\\thispagestyle{empty}\n\\newpage\n\\thispagestyle{empty}\n\\hypersetup{linkcolor=black}\n\\tableofcontents\n\\thispagestyle{empty}\n\\newpage')

    # fix pandoc overleaf issue
    body = body.replace('\\pandocbounded', '')

    # make section links work
    body = body.replace('→ Data Analysis', '\\ref{data-analysis}~\\nameref{data-analysis}')
    body = body.replace('→Data Cleaning/SMARD-Data Preprocessing', '\\ref{smard-data-preprocessing}~\\nameref{smard-data-preprocessing}')
    body = body.replace('→SMARD\nElectricity Data', '\\ref{smard-electricity-market-data}~\\nameref{smard-electricity-market-data}')
    body = body.replace('→SMARD Electricity\nData', '\\ref{smard-electricity-market-data}~\\nameref{smard-electricity-market-data}')
    body = body.replace('→ Gathering Domain Knowledge', '\\ref{gathering-domain-knowledge}~\\nameref{gathering-domain-knowledge}')
    body = body.replace('→ Data Sources', '\\ref{data-sources}~\\nameref{data-sources}')
    body = body.replace('→ Visualization \\& Story Telling', '\\ref{visualization-story-telling}~\\nameref{visualization-story-telling}')
    body = body.replace('→ Predictive Modelling', '\\ref{predictive-modeling}~\\nameref{predictive-modeling}')
    body = body.replace('→ Summary \\&\nFuture Work', '\\ref{summary-future-work}~\\nameref{summary-future-work}')

    # shorten TOC
    body = body.replace('\\subsection{Acknowledgement}','\\vspace{15em}\\subsection{Acknowledgement}')

    # set A4
    body = body.replace('\\documentclass[11pt]{article}','\\documentclass[a4paper]{article}')


    with open('output.tex', 'w', encoding='utf-8') as f:
        f.write(body)


    print("done")
except Exception as e:
    print(f"Error: {e}")

