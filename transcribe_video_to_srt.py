from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import subprocess
import os
import tempfile


def extract_audio(video_path):
    """Extract audio from video file"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        audio_path = temp_audio_file.name
    command = [
        "ffmpeg",
        "-y",  # Overwrite output files without asking
        "-i",
        video_path,
        "-ar",
        "16000",  # Set sample rate to 16kHz
        "-ac",
        "1",  # Convert to mono
        "-vn",  # Disable video recording
        audio_path,
    ]
    subprocess.run(command, check=True)
    return audio_path


def transcribe_video(video_path):
    # First extract audio from video
    audio_path = extract_audio(video_path)
    try:
        # Check if CUDA (GPU) is available
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load the model and processor
        model_id = "NbAiLab/nb-whisper-medium"

        # Initialize the model
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        # Load the model
        model.to(device)

        # Load the processor
        processor = AutoProcessor.from_pretrained(model_id)

        # Create the pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
            generate_kwargs={
                "task": "transcribe",
                "language": "no",
                "num_beams": 5,
                "max_new_tokens": 128,
            },
        )

        # Transcribe the audio
        result = pipe(audio_path)

        # Print and save the transcript in SRT format
        srt_content = []
        for i, segment in enumerate(result["chunks"], start=1):
            start = segment["timestamp"][0]
            end = (
                segment["timestamp"][1]
                if segment["timestamp"][1] is not None
                else start + 1
            )  # Handle missing end timestamp
            text = segment["text"]

            # Convert seconds to hours, minutes, seconds, and milliseconds
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                millis = int((seconds % 1) * 1000)
                return "{:02}:{:02}:{:02},{:03}".format(hours, minutes, secs, millis)

            start_time = format_time(start)
            end_time = format_time(end)

            srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

        srt_content = "".join(srt_content)
        print(srt_content)
        with open("transcript.srt", "w", encoding="utf-8") as f:
            f.write(srt_content)
    finally:
        os.remove(audio_path)  # Clean up the temporary audio file


# Replace with your video path
video_path = "video.mp4"
transcribe_video(video_path)
