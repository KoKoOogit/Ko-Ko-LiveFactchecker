import numpy as np
from pyannote.audio import Model, Inference, Pipeline
from faster_whisper import WhisperModel
from scipy.spatial.distance import cdist
from pyannote.core import Segment
from pydub import AudioSegment
from groq import Groq
import torch
import os
import uuid
from io import BytesIO
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
import json 

class AudioProcessor:
    def __init__(self):
        self.diarizationModel = "pyannote/speaker-diarization-3.1"
        self.diarizationPipeline = Pipeline.from_pretrained(
            self.diarizationModel,
            use_auth_token=os.environ['HF_TOKEN'])

        self.whisperModelSize = "base"
        self.whisperModel = WhisperModel(
            self.whisperModelSize, device="cpu", compute_type="int8")

        # Ensure you use the correct device
        self.diarizationPipeline.to(torch.device("cpu"))
        self.groqClient = Groq(
            api_key=os.environ['GROQ_API_KEY']
        )
        self.embeddingModel = Model.from_pretrained(
            "pyannote/embedding",
            use_auth_token=os.environ['HF_TOKEN']
        )
        self.embeddingInference = Inference(
            self.embeddingModel, window="whole")
        self.embeddings = []
        self.superEmbeddings = {}
        self.active_speaker  = -1
        self.threshold = 0.35
        self.counter = 0
  
    def groqTranscribe(self, wav_file):
        with open(wav_file, "rb") as f:
            transcription = self.groqClient.audio.transcriptions.create(
                file=(wav_file, f.read()),  # Required audio file
                model="distil-whisper-large-v3-en",  # Required model to use for transcription
                prompt="This contexts is a debate",  # Optional
                response_format="json",  # Optional
                language="en",  # Optional
            )
            return transcription.text

    def localTranscribe(self, wav_file):
        segments, info = self.whisperModel.transcribe(wav_file, beam_size=5)
        text = "".join([segment.text for segment in segments])
        return text

    def embed_chunk(self, wav_file):
        """Extracts embeddings for a given wav file."""
        emb = self.embeddingInference(wav_file)
        if len(self.embeddings) == 0:  # Initialize the embeddings if empty
            self.embeddings.append({
                "speaker": 0,
                "data": emb
            })
        return emb

    def compare_embedding(self, wav_file):
        """Compare embeddings to find the closest speaker."""
        emb = self.embed_chunk(wav_file)
        min_distance = float("inf")
        possible_speaker = None

        if self.embeddings:  # Check if there are any existing embeddings
            for i in self.embeddings:
                dist = cdist(emb.reshape(1, -1),
                             i['data'].reshape(1, -1), metric="cosine")[0][0]
                if dist < min_distance:
                    possible_speaker = i['speaker']
                    min_distance = dist

            if min_distance > self.threshold:
                new_speakerID = len(self.embeddings)
                self.embeddings.append(
                    {
                        "speaker": new_speakerID,
                        "data": emb
                    }
                )
                return new_speakerID
        return possible_speaker
    
    # Function to convert numpy array to pydub AudioSegment


    def np_array_to_audiosegment(self,np_array, sample_rate=44100, channels=1, sample_width=2):
        """
        Converts numpy array to pydub AudioSegment.

        :param np_array: numpy array of audio samples
        :param sample_rate: Sample rate of the audio (default 44100)
        :param channels: Number of channels (default 1)
        :param sample_width: Number of bytes per sample (default 2 for int16)
        :return: pydub.AudioSegment object
        """
        # Ensure the array is in the correct shape for multi-channel audio
        if channels > 1:
            np_array = np_array.flatten()

        # Convert numpy array to bytes
        audio_data = np_array.astype(np.int16).tobytes()

        # Create pydub AudioSegment from raw data
        return AudioSegment(
            data=audio_data,
            sample_width=sample_width,
            frame_rate=sample_rate,
            channels=channels
        )

    def write_to_file(self, filename, content):
        """Writes content to a file, appending if the file exists.

        Args:
            filename: The name of the file to write to.
            content: The content to write to the file.
        """

        mode = 'w'  # Write mode (create new file if it doesn't exist)
        if os.path.exists(filename):
            mode = 'a'  # Append mode (add content to existing file)

        with open(filename, mode) as f:
            f.write(content)


    def start_process_from_file(self, wav_file):
        """Processes the audio file to determine speakers."""
        # Ensure temp directory exists
        os.makedirs("temp", exist_ok=True)
       
        segments, info = self.whisperModel.transcribe(wav_file, beam_size=5)
        visited_speakers = set()
        active_file = AudioSegment.from_file(wav_file)
        for segment in segments:
            if segment.id not in visited_speakers:
                uid = uuid.uuid4()
                excerpt = active_file[segment.start * 1000: segment.end * 1000]
                filename = f"temp/{uid}.wav"
                if excerpt.duration_seconds > 1:
                   
                   
                    excerpt.export(filename, format="wav")
                    predictedSpeaker = self.compare_embedding(filename)
                    if self.active_speaker != -1 and predictedSpeaker != self.active_speaker:
                        #feed it to LLM
                        pass
                    self.write_to_file("transcriptions.txt", f"Speaker {predictedSpeaker}: {segment.text} \n")
                    visited_speakers.add(segment.id)
                    os.remove(filename)

        return self.embeddings

    async def start_process_from_nparr(self, nparr, tp, websocket):
        """Processes the audio file to determine speakers."""
        # Ensure temp directory exists
        os.makedirs("temp", exist_ok=True)
        active_file = self.np_array_to_audiosegment(nparr)
        active_file.export("weboutput.mp3")
        segments, info = self.whisperModel.transcribe("weboutput.mp3", beam_size=5)
        visited_speakers = set()
        
        for segment in segments:
            transcript_obj = {
                "mode":"transcript",
                "data":segment.text
            }
            await websocket.send(json.dumps(transcript_obj))
           

            if segment.id not in visited_speakers:
                uid = uuid.uuid4()
                excerpt = active_file[segment.start * 1000: segment.end * 1000]
                filename = f"temp/{uid}.wav"
                if excerpt.duration_seconds > 1:

                    excerpt.export(filename, format="wav")
                    predictedSpeaker = self.compare_embedding(filename)
                    if self.active_speaker != -1 and predictedSpeaker != self.active_speaker:
                        llm_response = tp.process()
                        self.active_speaker = predictedSpeaker
                        obj = {"type":"llm_response", "data":llm_response}
                        await websocket.send(json.dumps(obj))
                       
                    
                    if self.active_speaker == -1:
                        self.active_speaker = predictedSpeaker
                        
                    self.write_to_file("transcriptions.txt", f"Speaker {
                                  predictedSpeaker}: {segment.text} \n")
                    visited_speakers.add(segment.id)
                    os.remove(filename)

        return self.embeddings

    async def get_llm_output(self, nparr, tp, websocket):
        
        os.makedirs("temp", exist_ok=True)
        active_file = self.np_array_to_audiosegment(nparr)
        active_file.export("weboutput.mp3")
        diarization = self.diarizationPipeline(
            file="weboutput.mp3", max_speakers=5)
               
        for turn, _, speaker in diarization.itertracks(yield_label=True):
           
            print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            turn_chunk = active_file[turn.start*1000:turn.end*1000]
            uid = uuid.uuid4()
            chunk_filename = f"temp/{uid}.wav"
            turn_chunk.export(chunk_filename,format="wav")
            transcriptions = self.localTranscribe(chunk_filename)
            if transcriptions != "":
                self.write_to_file("transcriptions.txt", f"speaker_{speaker}: {transcriptions} \n")
                transcript_obj = {
                    "mode": "transcript",
                    "data": transcriptions
                }
                await websocket.send(json.dumps(transcript_obj))

        #feed to llm
        llm_response = tp.process()
        if llm_response != "" and llm_response != []:
            llm_response_string = json.dumps(llm_response)
            llm_response_object = {
                "mode": "llm_response", "data": llm_response_string} 
            await websocket.send(json.dumps(llm_response_object))
            print("============ Work Done ================")