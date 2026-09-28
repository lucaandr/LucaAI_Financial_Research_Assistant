from langchain_community.embeddings import HuggingFaceEmbeddings

print(" all-MiniLM-L6-v2...")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

sample_text = "Micron reported Q3 revenue of $6.81 billion."
query_result = embeddings.embed_query(sample_text)

print("\nSucces!")
print(f"Dimensiunea vectorului generat: {len(query_result)}")
print(f"Primele 5 valori din vector: {query_result[:5]}")
