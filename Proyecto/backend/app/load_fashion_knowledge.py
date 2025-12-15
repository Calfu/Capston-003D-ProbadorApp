import chromadb

chroma = chromadb.Client()
collection = chroma.get_or_create_collection("probador_fashion")

docs = [
    {
        "id": "colores_basicos",
        "text": """
Reglas de color:
- Negro combina con todo.
- Beige + café = look moderno.
- Gris + negro = minimalista.
- Verde oliva combina con beige, café y negro.
"""
    },
    {
        "id": "outfits",
        "text": """
Guía de outfits:
- Cargo café + polera blanca + zapatillas blancas.
- Chaqueta bomber + jeans rectos + polera lisa.
- Polera beige + pantalón negro = look limpio.
"""
    },
    {
        "id": "cuerpo",
        "text": """
Guía para tipos de cuerpo:
- Personas bajas: evitar prendas muy largas.
- Atléticos: prendas rectas o slim.
"""
    }
]

for doc in docs:
    collection.add(
        documents=[doc["text"]],
        ids=[doc["id"]]
    )

print("Fashion DB cargada.")