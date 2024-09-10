import asyncio
import websockets
import numpy as np
from AudioProcessor import AudioProcessor
from TextProcessor import TextProcessor
import io
import wave
# Function to handle incoming WebSocket connections
ap = AudioProcessor()
tp = TextProcessor()
chunks = np.array([], dtype=np.int16)


async def receive_audio(websocket, path):
    global chunks
    print("Client connected")
    try:
        await websocket.send("Hi!")
        while True:
            # Receive the raw audio data sent from the client
            data = await websocket.recv()
            if  isinstance(data,str):
                filename = data+".wav"
                with wave.open(filename, 'w') as wf:
                    # Set parameters for the wave file
                    n_channels = 1      # Mono
                    sampwidth = 2       # Sample width in bytes (16-bit)
                    framerate = 44100   # Frame rate (samples per second)
                    n_frames = 0        # Number of frames (samples)
                    comptype = 'NONE'   # No compression
                    compname = 'not compressed'

                    # Set the parameters
                    wf.setnchannels(n_channels)
                    wf.setsampwidth(sampwidth)
                    wf.setframerate(framerate)
                    wf.setnframes(n_frames)
                    wf.setcomptype(comptype, compname)
            elif isinstance(data,bytes):
                session_id_length = 18
                # Convert the received data (bytes) to numpy array
                combined_buffer = np.frombuffer(data, dtype=np.int16)
                # Extract sessionID Int16 array and decode it to a string
                session_id_int16 = combined_buffer[:session_id_length]
                session_id_bytes = session_id_int16.astype(np.uint8).tobytes()
               
                session_id = session_id_bytes.decode('utf-8')
               
                audio_array = combined_buffer[session_id_length:]
                # Concatenate the new audio data with existing chunks
                chunks = np.concatenate((chunks, audio_array))

                # Calculate the duration of the audio data in seconds
                duration = len(chunks) / 44100

                # Check if the duration is approximately 10 seconds
                if duration >=5:
                    await ap.start_process_from_nparr(chunks,tp=tp, websocket=websocket)
                    print("Processed 10-second audio data:", chunks)
                    chunks = np.array([], dtype=np.int16)  # Reset chunks
            else:
                print("Some kind of weird data!")
            # Print the received audio data for debugging
            # print("Received audio data:", audio_array)
    except websockets.exceptions.ConnectionClosed as e:
        print("Client disconnected", e)
 
# Start WebSocket server


async def start_server():
    print("Starting server...")
    server = await websockets.serve(receive_audio, "localhost", 6789)
    await server.wait_closed()

# Run the server
asyncio.run(start_server())
