
def create_chunks(
    sentences: list[dict],
    max_duration: float = 60.0
) -> list[dict]:
    chunks = []
    current = []

    for sentence in sentences:
        if not current:
            current.append(sentence)
            continue

        current_start = current[0]["start"]
        new_end = sentence["end"]
        duration = new_end - current_start

        if duration <= max_duration:
            current.append(sentence)
        else:
            chunks.append({
                "start": round(current[0]["start"], 2),
                "end": round(current[-1]["end"], 2),
                "text": " ".join(s["text"] for s in current),
                "sentences": current
            })
            current = [sentence]

    if current:
        chunks.append({
            "start": round(current[0]["start"], 2),
            "end": round(current[-1]["end"], 2),
            "text": " ".join(s["text"] for s in current),
            "sentences": current
        })

    return chunks






# def create_chunks(
#     sentences: list[dict],
#     max_duration: float = 60
# ) -> list[dict]:

#     chunks = []

#     current = []

#     for sentence in sentences:

#         if not current:

#             current.append(sentence)
#             continue

#         current_start = current[0]["start"]
#         new_end = sentence["end"]

#         duration = new_end - current_start

#         if duration <= max_duration:

#             current.append(sentence)

#         else:

#             chunks.append({
#                 "start": current[0]["start"],
#                 "end": current[-1]["end"],
#                 "text": " ".join(
#                     s["text"] for s in current
#                 )
#             })

#             current = [sentence]

#     if current:

#         chunks.append({
#             "start": current[0]["start"],
#             "end": current[-1]["end"],
#             "text": " ".join(
#                 s["text"] for s in current
#             )
#         })

#     return chunks