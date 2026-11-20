from src.chunk import create_chunks, generate_chunk_id
def test_create_chunks():
    text = " ".join(["python"] * 1000)
    chunks = create_chunks(
        text,
        chunk_size=100,
        overlap=20
    )
    assert len(chunks) == 13
    assert len(chunks[0].split()) == 100
def test_generate_chunk_id():
    first = generate_chunk_id("document.pdf",1,1,"Python document")
    second = generate_chunk_id("document.pdf",1,1,"Python document")
    assert first == second
