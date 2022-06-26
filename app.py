from flask_cors import CORS
from flask import Flask
from flask import request , jsonify

app = Flask(__name__)

CORS(app)

#import important library for Application
import clip
import torch
from io import BytesIO
import requests
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# Load the photo IDs
photo_ids = pd.read_csv("unsplash-dataset/photo_ids.csv")
photo_ids = list(photo_ids['photo_id'])

# Load the features vectors
photo_features = np.load("unsplash-dataset/features.npy")

# Convert features to Tensors: Float32 on CPU and Float16 on GPU
if device == "cpu":
  photo_features = torch.from_numpy(photo_features).float().to(device)
else:
  photo_features = torch.from_numpy(photo_features).to(device)

# Print some statistics
print(f"Photos loaded: {len(photo_ids)}")



def encode_search_query(search_query):
  with torch.no_grad():
    # Encode and normalize the search query using CLIP
    text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
    text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
  # Retrieve the feature vector
  return text_encoded

def find_best_matches(text_features, photo_features, photo_ids, results_count=3):
  # Compute the similarity between the search query and each photo using the Cosine similarity
  similarities = (photo_features @ text_features.T).squeeze(1)
  # print(len(similarities))
  # print(max(simil arities))
  # Sort the photos by their similarity score
  best_photo_idx = (-similarities).argsort()
  # Return the photo IDs of the best matches
  return [photo_ids[i] for i in best_photo_idx[:results_count]]

def search_unslash(search_query, photo_features, photo_ids, results_count=3):
  # Encode the search query
  text_features = encode_search_query(search_query)

  # Find the best matches
  best_photo_ids= [] # for prediction populirty 
  best_photo_urls = []
  best_photo_ids = find_best_matches(text_features, photo_features, photo_ids, results_count)   
  for id in   best_photo_ids  : 
    best_photo_urls.append(f"https://unsplash.com/photos/{id}/download")
    print(f"https://unsplash.com/photos/{id}/download")
  return best_photo_urls 

# model.avgpool = nn.AdaptiveAvgPool2d(1) # for any size of the input
model_pred = torchvision.models.resnet50()
model_pred.fc = torch.nn.Linear(in_features=2048, out_features=1)
model_pred.load_state_dict(torch.load('model/model-resnet50.pth', map_location=device)) 
model_pred.eval().to(device)


def prepare_image(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    Transform = transforms.Compose([
            transforms.Resize([224,224]),      
            transforms.ToTensor(),
            ])
    image = Transform(image)   
    image = image.unsqueeze(0)
    return image.to(device)




def predict(image, model_pred):
    with torch.no_grad():
        preds = model_pred(image)
    
    return preds.item()


@app.route("/search" , methods=['GET' , 'POST'])
def search():
  text =  "Sydney Opera house" if  request.args.get("text") is  None else request.args["text"]
  num = 3 if  request.args.get("num") is  None else int(request.args["num"])
  print(text , "   " , len(text) , "   " , type(text) ) 
  images  = search_unslash(text, photo_features, photo_ids,num)
  return jsonify(images = images )

import requests
#@app.route("/predict", methods=['GET' , 'POST'])
#def predicting():
     #try:
      #    image_url = request.args["image_url"]
     #     img = Image.open(BytesIO(requests.get(image_url).content))
     #except:
      #    img = image_url = Image.open(request.files["image_url"].stream)
     #print(type(image_url))
     #response = requests.get(image_url)
#     print(response)
     ##img = Image.open(BytesIO(response))
     #image = prepare_image(img)
     #x = predict(image,model_pred)
     #if x < -2:
     #   x = 0
     #elif x > 8 :
     #   x = 8 
     #x = f"{(2+int(x)) * 10 } %"
     #return jsonify(predect_result = x)
@app.route("/predict" ,  methods=['GET' , 'POST'])
def gg():
     imagess = {}
     # print("gggg" , len(request.files("image")))
     if  request.args.getlist("image_url") != None :
          for item in request.args.getlist("image_url"):
               imagess[item] = { "content" :Image.open(BytesIO(requests.get(item).content)) , "result": 0}
     if  request.files.getlist("image_url")  != None :
          for item in  request.files.getlist("image_url"):
               imagess[item.filename] = { "content" :Image.open(item.stream) , "result": 0}
               #images.update(Image.open(item).stream)
     # image = prepare_image(img
     #print(imagess)
     for img in imagess:
          image = prepare_image(imagess[img]["content"])
          x = predict(image,model_pred)
          if x < -2:
               x = 0
          elif x > 8 :
               x = 8 
          x = f"{(2+int(x)) * 10 } %"
          imagess[img]["result"] =  x
          del imagess[img]["content"]
     print(imagess)
     return jsonify(resultOfPredicting	 = imagess)
