import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

models = client.models.list()
print("Modele disponibile in contul tau Groq:")
for model in models.data:
    print(f"- {model.id}")

target_model = models.data[0].id
print(f"\nTestam modelul: {target_model}")

completion = client.chat.completions.create(
    model=target_model,
    messages=[{"role": "user", "content": "Salut! Conexiunea functioneaza?"}],
)

print("\nRaspuns de la Groq:")
print(completion.choices[0].message.content)
