from nbconvert import LatexExporter

notebook_path = 'report.ipynb'

try:
    latex_exporter = LatexExporter()
    body, resources = latex_exporter.from_filename(notebook_path)

    body = body.replace('\\begin{longtable}[]{@{}', "\\rowcolors{2}{white}{gray!25}\n\\begin{longtable}[]{@{}")
    body = body.replace('\\usepackage{graphicx}', '\\usepackage[table]{xcolor}\n\\usepackage{graphicx}')
    body = body.replace('\\title{report}','\\title{BTW 2025 Data Science Challenge}')
    body = body.replace('\\section{BTW 2025 Data Science\nChallenge}\\label{btw-2025-data-science-challenge}', '')
    body = body.replace('\\subsection{', '\\section{')
    body = body.replace('\\subsubsection{', '\\subsection{')
    body = body.replace('>{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.1221}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.0560}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.0840}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.7379}}@{}}', \
                        '>{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.25}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.15}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.2}}\n  >{\\raggedright\\arraybackslash}p{(\\linewidth - 6\\tabcolsep) * \\real{0.4}}@{}}')
    body = body.replace('\\pandocbounded{\\includegraphics[keepaspectratio]{src/attention_graphic.png}}',\
                        '\\raisebox{-0.8\\height}{\\includegraphics[keepaspectratio]{src/attention_graphic.png}}')

    startutils = body.index('\\section{Utils}\\label{utils}') + len('\\section{Utils}\\label{utils}')
    endutils = body.index('\\section{Gathering Domain')
    body = body[:startutils] + '\\textit{code omitted - see notebook}' + body[endutils:]

    with open('output.tex', 'w', encoding='utf-8') as f:
        f.write(body)


    print("done")
except Exception as e:
    print(f"Error: {e}")

