#convert the videos to mp3
import os
import subprocess

files = os.listdir("videos")
for file in files:
    print(file)
    # Split by "Tutorial #" and get the part after it
    if "Tutorial #" in file:
        tutorial_number = file.split("Tutorial #")[1].split(".")[0]
        print(tutorial_number)
        file_name = file.split("_")[0].strip()
        print(tutorial_number, file_name)
        subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{tutorial_number}_{file_name}.mp3"])
    else:
        print(f"Skipping {file} - doesn't match expected format")