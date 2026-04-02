from app.services.pdf_processor import process_pdf

docs = process_pdf("papers/Bias_bounties.pdf")

print(len(docs))
print(docs[0])