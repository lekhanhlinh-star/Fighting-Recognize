from fastapi.responses import JSONResponse
import uvicorn
from fastapi import FastAPI, UploadFile, File
import cv2
from tools.convert_PIL_base64 import base64_to_pil_image,pil_image_to_base64
from pytorchvideo.data.encoded_video import EncodedVideo
from classifier.model import clip_duration,transform,torch,load_model
from fastapi import FastAPI, WebSocket,WebSocketDisconnect
from fastapi.responses import HTMLResponse,RedirectResponse
import os
import tempfile

app = FastAPI()
start_sec = 0
import numpy as np
end_sec = start_sec + clip_duration
model_classifier =load_model().to("cuda:0")
model_classifier.eval()

class MyVideo(object):
    def __init__(self):
        self.stack = []
    def to_tensor(self, img):
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img.unsqueeze(0)
    def get_video_clip(self):
        assert len(self.stack) > 0, "clip length must large than 0 !"
        self.stack = [self.to_tensor(img) for img in self.stack]
        clip = torch.cat(self.stack).permute(-1, 0, 1, 2)
        del self.stack
        self.stack = []
        return clip
    def save_frames(self,image):
        self.stack.append(image)

video=MyVideo()
@app.get("/")
async def read_root():
    return {"Hello": "World"}
# @app.post("/upload/")
# async def upload_mp4_file(file: UploadFile = File(...)):
#     # Save the uploaded file to a temporary location
#     file_path = f"temp_{file.filename}"
#     with open(file_path, "wb") as temp_file:
#         temp_file.write(file.file.read())
    
#     # Open the saved file using OpenCV
#     video=EncodedVideo.from_path(file_path)
#     video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
#     video_data = transform(video_data)
#     inputs = video_data["video"]
    
#     inputs = [i[None, ...].to("cuda:0") for i in inputs]
    
#     with torch.no_grad():
#         preds = model_classifier(inputs)

#         # Get the predicted classes
#         post_act = torch.nn.Sigmoid()
#         preds = post_act(preds)
#         labels_names=["Fight","Normal"]
#         print(preds)
#         print(labels_names[preds[0]>0.5])
   
    
#     # Process the video or perform any necessary operations
    
#     # Clean up: Close the video capture and delete the temporary file
   
#     os.remove(file_path)
    
#     return JSONResponse(content={"message": labels_names[preds[0]>0.5]})
@app.websocket("/upload/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
# async def upload_mp4_file(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    data = await websocket.receive_bytes()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as temp_file:
        temp_file.write(data)
        temp_file_path = temp_file.name

    # Open the saved file using OpenCV
    video=EncodedVideo.from_path(temp_file_path)
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
    video_data = transform(video_data)
    inputs = video_data["video"]
    
    inputs = [i[None, ...].to("cuda:0") for i in inputs]
    
    with torch.no_grad():
        preds = model_classifier(inputs)

        # Get the predicted classes
        post_act = torch.nn.Sigmoid()
        preds = post_act(preds)
        labels_names=["Fight","Normal"]
        print(preds)
        print(labels_names[preds[0]>0.5])
        await websocket.send_text(labels_names[preds[0]>0.5])
    os.remove(temp_file_path)
    return JSONResponse(content={"message": labels_names[preds[0]>0.5]})

@app.websocket("/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            img=base64_to_pil_image(data.split(",")[1])
            img=np.array(img)
            video.save_frames(image=img)
            if len(video.stack)==25:
                # print(f"processing {cap.idx // 25}th second clips")
                clip = video.get_video_clip()
                data_clip={"video":clip}
                video_data=transform(data_clip)
                inputs = video_data["video"]
                inputs = [i[None, ...].to("cuda:0") for i in inputs]
                with torch.no_grad():
                    preds = model_classifier(inputs)
                    # Get the predicted classes
                    post_act = torch.nn.Sigmoid()
                    preds = post_act(preds)
                    labels_names=["Fight","Normal"]
                    print(preds)
                    print(labels_names[preds[0]>0.5])
                    await websocket.send_text(labels_names[preds[0]>0.5])
        except WebSocketDisconnect:
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001)