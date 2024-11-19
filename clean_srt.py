import re


def clean_srt(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove subtitle numbers, timestamps, and HTML tags
    cleaned = re.sub(
        r"\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n", "", content
    )
    cleaned = re.sub(r"<[^>]+>", "", cleaned)

    # Remove empty lines
    cleaned = "\n".join(line for line in cleaned.split("\n") if line.strip())

    print(cleaned)


clean_srt("input.srt")
