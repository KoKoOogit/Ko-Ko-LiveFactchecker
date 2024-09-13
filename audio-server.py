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
        while True:
            # Receive the raw audio data sent from the client
            data = await websocket.recv()
            if  isinstance(data,str):
                if len(data) > 8 and data[:8] == "speakers":
                    tp.set_speakers(data)

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
                if duration >=10:
                    await ap.get_llm_output(nparr=chunks,tp=tp, websocket=websocket)
                    
                    chunks = np.array([], dtype=np.int16)  # Reset chunks
                    
            else:
                print("Some kind of weird data!")

    except websockets.exceptions.ConnectionClosed as e:
        print("Client disconnected", e)
 
# Start WebSocket server


async def start_server():
    print("Starting server...")
    server = await websockets.serve(receive_audio, "localhost", 6789,ping_interval=None)
    await server.wait_closed()

# Run the server
asyncio.run(start_server())
