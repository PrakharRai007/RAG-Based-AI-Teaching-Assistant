import whisper
import json
import os

model = whisper.load_model("base")  # Fast and good for testing

audios = os.listdir("audios")

# Create json directory if it doesn't exist
if not os.path.exists("json"):
    os.makedirs("json")

for audio in audios:
    if audio.endswith(".mp3"):
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]  # Remove .mp3 extension
        print(number, title)

        result = model.transcribe(audio=f"audios/{audio}",  # ✅ Changed - removed the extra .mp3
                                 language="hi",
                                 task="translate",
                                 word_timestamps=False)
        
        chunks = []
        for segment in result["segments"]:
            chunks.append({"number": number, "title": title, "start": segment["start"], "end": segment["end"], "text": segment["text"]})
        
        chunks_with_metadata = {"chunks": chunks, "text": result["text"]}
        
        # Remove .mp3 extension before adding .json
        filename = audio[:-4]  # Remove .mp3
        with open(f"json/{filename}.json", "w") as f:
            json.dump(chunks_with_metadata, f)