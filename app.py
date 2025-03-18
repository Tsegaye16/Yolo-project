import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import torch
import os
from PIL import Image
import numpy as np
import base64
import io
from ultralytics import YOLO

# Ensure the model path is correct
MODEL_PATH = os.path.join("runs", "detect", "train", "weights", "best.pt")

# Check if model file exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please check the path.")

# Load the YOLO model and move it to CPU
model = YOLO(MODEL_PATH)
model.to("cpu")

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define the layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("YOLO Object Detection", className="text-center my-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    html.I(className="fas fa-upload me-2"),
                    "Drag and Drop or ",
                    html.A("Select an Image", className="text-primary")
                ]),
                style={
                    'width': '100%',
                    'height': '150px',
                    'lineHeight': '150px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '10px',
                    'textAlign': 'center',
                    'cursor': 'pointer',
                    'backgroundColor': '#f8f9fa',
                    'color': '#6c757d',
                    'fontSize': '18px'
                },
                multiple=False
            ),
        ], width=12, className="mb-4")
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='output-image-upload'), width=12)
    ])
], fluid=True, style={'padding': '20px'})

# Function to parse the uploaded image
def parse_image(contents):
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded)).convert("RGB")  # Convert to RGB
        return np.array(image)
    except Exception as e:
        print(f"Error parsing image: {e}")
        return None

# Callback to handle image upload and prediction
@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'))
def update_output(contents):
    if contents is None:
        return dbc.Alert("Please upload an image.", color="info", className="my-4")

    # Parse the image
    image = parse_image(contents)
    if image is None:
        return dbc.Alert("Error loading image.", color="danger", className="my-4")

    # Perform prediction
    results = model(image)
    
    # Debugging: Print results
    print("Model output:", results)

    if not hasattr(results[0], 'boxes'):
        return dbc.Alert("Error: Model output is not in the expected format.", color="danger", className="my-4")

    # Get the detected objects
    detections = results[0].boxes.data.cpu().numpy()
    analysis = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if conf < 0.65:
            label = "Unknown"
        else:
            label = results[0].names[int(cls)]
        analysis.append(f"{label} with confidence {conf:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    # Plot the detections
    plotted_image = results[0].plot()

    # Convert to PIL image
    plotted_image_pil = Image.fromarray(plotted_image[..., ::-1])  # Convert BGR to RGB

    # Resize the image to fit properly
    max_width = 800
    original_width, original_height = plotted_image_pil.size
    aspect_ratio = original_height / original_width
    new_height = int(max_width * aspect_ratio)
    resized_image = plotted_image_pil.resize((max_width, new_height), Image.Resampling.LANCZOS)

    # Convert the image to base64
    buffered = io.BytesIO()
    resized_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    print(f"Generated base64 image length: {len(img_str)}")

    # Return the results
    return dbc.Card([
        dbc.CardHeader(html.H4("Detection Results", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.Img(src=f"data:image/png;base64,{img_str}",
                                 style={'width': '100%', 'height': 'auto', 'borderRadius': '10px'}),
                        width=6),
                dbc.Col([
                    html.H4("Detection Analysis:", className="mt-3"),
                    html.Ul([html.Li(item, className="mb-2") for item in analysis], className="list-unstyled")
                ], width=6)
            ])
        ])
    ], className="my-4")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
