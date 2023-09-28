from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from predict import read_image, preprocess_image, load_model, generate_prediction, top_matches
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Define a list of allowed origins (domains)
# Replace '*' with the specific origins you want to allow, e.g., ['https://example.com', 'https://another-domain.com']
allowed_origins = ["https://bee-api-ng8o.onrender.com"]

# Add CORS middleware to allow certain URLs access
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,  # Allow cookies in cross-origin requests
    allow_methods=["*"],     # You can specify specific HTTP methods if needed
    allow_headers=["*"],     # You can specify specific headers if needed
)

@app.get('/')
def root_hello():
    return {'Hello': 'World'}

@app.post('/generate')
async def predict_image(file: UploadFile = File(...)):
    #Read file
    image = read_image(await file.read())
    #Preprocess image
    image = preprocess_image(image)
    #Predict
    model = load_model()
    predictions = generate_prediction(model, image)
    top_names, top_probabilities = top_matches(predictions)

    top_names_norm = []
    for i in top_names:
        top_names_norm.append(str(i))

    top_probabilities_norm = []
    for i in top_probabilities:
        top_probabilities_norm.append(i.item())

    result = list(zip(top_names_norm, top_probabilities_norm))
    
    return result




# @app.get('/show')
# def show_image():
#     return FileResponse('path.jpg')
