
# first create a list of all pdfs

import pdfplumber
import os
target_folder="D:\\Office\\ocr_jobs\\2023-24 Transfer from"
pdfs=[f for f in os.listdir(target_folder) if f.endswith('.pdf')]

print(pdfs)

# for all pdfs, extract those tables which are in annexure II and preesent them in single csv which will have data from all pdfs

all_tables = []

for pdf in pdfs:
    print(f"Processing {pdf}")
    with pdfplumber.open(os.path.join(target_folder,pdf)) as pdf_doc:
        for i, page in enumerate(pdf_doc.pages):
            text = page.extract_text()
            if text and "Annexure II" in text:
                print(f"Found Annexure II on page {i+1} of {pdf}")
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    print(f"Table {t_idx+1} on page {i+1} of {pdf}:")
                    for row in table:
                        print(row)
                    # Save table to CSV
                    all_tables.extend(table)
import pandas as pd
# Convert all extracted tables to a single DataFrame and save as CSV
if all_tables:
    df = pd.DataFrame(all_tables[1:], columns=all_tables[0])
    output_csv = os.path.join(target_folder, "extracted_tables_annexure_II.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved all extracted tables to {output_csv}")