import whisper
import json

model = whisper.load_model("base")  # Fast and good for testing

result = model.transcribe(audio="audios/sample.mp3.mp3", language="hi", task="translate", word_timestamps=False)

print(result["segments"])
chunks = []
for segment in result["segments"]:
    chunks.append({"start": segment["start"], "end": segment["end"], "text": segment["text"]})

print(chunks)

with open("output.json", "w") as f:
    json.dump(chunks,f)