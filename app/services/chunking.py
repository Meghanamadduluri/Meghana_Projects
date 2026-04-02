from nltk.tokenize import sent_tokenize

def chunk_text(text: str, max_words: int = 120):

    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:

        words = sentence.split()

        if word_count + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            word_count = 0

        current_chunk.append(sentence)
        word_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks