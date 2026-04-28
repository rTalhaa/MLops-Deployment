import time
import random
import requests

API_URL = "http://localhost:8000/predict"

smiles_examples = [
    "CCO",
    "CCCl",
    "CCN",
    "C1=CC=CC=C1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCBr",
    "CC(C)O",
    "c1ccccc1O",
    "CCN(CC)CC",
    "INVALID_SMILES"
]

for i in range(100):
    smiles = random.choice(smiles_examples)

    try:
        response = requests.post(
            API_URL,
            json={"smiles": smiles},
            timeout=5
        )

        print(i + 1, smiles, response.status_code, response.text)

    except Exception as error:
        print("Request failed:", error)

    time.sleep(0.5)