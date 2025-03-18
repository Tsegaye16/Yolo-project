import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import torch
from PIL import Image
import numpy as np
import base64
import io
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO('runs/detect/train/weights/best.pt')

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))

    # Convert image to RGB if it has an alpha channel (4 channels)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    return np.array(image)

# Callback to handle image upload and prediction
@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'))
def update_output(contents):
    if contents is None:
        return dbc.Alert("Please upload an image.", color="info", className="my-4")

    # Parse the image
    image = parse_image(contents)

    # Perform prediction
    results = model(image)

    # Debug: Print the results object
    print(results)

    # Check if results is valid
    if not hasattr(results[0], 'boxes'):
        return dbc.Alert("Error: Model output is not in the expected format.", color="danger", className="my-4")

    # Plot the image with detections
    plotted_image = results[0].plot()  # This returns a NumPy array with detections drawn

    # Convert the plotted image to PIL format
    plotted_image_pil = Image.fromarray(plotted_image[..., ::-1])  # Convert BGR to RGB

    # Resize the image to a maximum width of 800 pixels (maintains aspect ratio)
    max_width = 800
    original_width, original_height = plotted_image_pil.size
    aspect_ratio = original_height / original_width
    new_height = int(max_width * aspect_ratio)
    resized_image = plotted_image_pil.resize((max_width, new_height), Image.Resampling.LANCZOS)

    # Convert PIL image to base64 for displaying in Dash
    buffered = io.BytesIO()
    resized_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Parse results for analysis
    detections = results[0].boxes.data.cpu().numpy()  # Extract bounding box data
    analysis = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if conf < 0.65:
            label = "Unknown"
        else:
            label = results[0].names[int(cls)]  # Get class name from model
        analysis.append(f"{label} with confidence {conf:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    # Display the image with detections and analysis
    return dbc.Card([
        dbc.CardHeader(html.H4("Detection Results", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.Img(src=f"data:image/png;base64,{img_str}", style={'width': '100%', 'height': 'auto', 'borderRadius': '10px'}), width=6),
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
