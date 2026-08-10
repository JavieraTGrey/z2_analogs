import PyPDF2
import glob

# Change the outdir and save pdf whenever you have to join results

#out_dir = r'/home/benjamin/Documents/analogs/dust/dcorr_pypeit_*.pdf'

#save_pdf = '/home/benjamin/Documents/analogs/results/dust_correction/dust_results_pypeit.pdf'

out_dir = r'/home/benjamin/Documents/analogs/HeII*.pdf'

save_pdf = '/home/benjamin/Documents/analogs/HeII4686.pdf'

results = sorted(glob.glob(out_dir))
print('Results read successfully.')


def PDFmerge(pdfs, output):
    # creating pdf file merger object
    pdfMerger = PyPDF2.PdfMerger()
  
    # appending pdfs one by one
    for pdf in pdfs:
        pdfMerger.append(pdf)
  
    # writing combined pdf to output pdf file
    with open(output, 'wb') as f:
        pdfMerger.write(f)
    print('Joint .pdf saved at:', output)


if __name__ == "__main__" :
    print('Saving joint pdf...')
    PDFmerge(results, save_pdf)