# Video to SRT Transcription

This script extracts audio from a video file and transcribes it into an SRT (SubRip Subtitle) file using a pre-trained speech recognition model.

## Dependencies

- Python 3.7+
- `transformers` library
- `torch` library
- `ffmpeg`

## Installation

1. Install `ffmpeg`:
    ```sh
    sudo apt-get install ffmpeg
    ```

2. Install Python dependencies:
    ```sh
    pip install transformers torch
    ```

## Usage

1. Place your video file in the same directory as the script or provide the path to the video file.
2. Run the script:
    ```sh
    python transcribe_video_to_srt.py
    ```

3. The transcript will be saved as `transcript.srt` in the same directory.

## Script Details

- `extract_audio(video_path)`: Extracts audio from the given video file and saves it as a temporary WAV file.
- `transcribe_video(video_path)`: Transcribes the extracted audio using the `NbAiLab/nb-whisper-medium` model and saves the transcript in SRT format.

## Example

Replace `video.mp4` with the path to your video file:

```python
video_path = "video.mp4"
transcribe_video(video_path)
```
