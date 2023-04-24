import dash
import dash_auth
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plot_common
import json
from shapes_to_segmentations import (
    compute_segmentations,
    blend_image_and_classified_regions_pil,
)
from skimage import io as skio
from skimage.transform import resize
from trainable_segmentation import multiscale_basic_features
import io
import base64
import PIL.Image
from PIL import ImageOps
import pickle
import shutil 
from time import time
from joblib import Memory
from credentials import VALID_USERNAME_PASSWORD_PAIRS

import os
from glob import glob
import numpy as np
import rasterio

memory = Memory("./joblib_cache", bytes_limit=3000000000, verbose=3)

compute_features = memory.cache(multiscale_basic_features)

DEFAULT_STROKE_WIDTH = 3  # gives line width of 2^3 = 8

EXAMPLE_IMAGE_PATH = "assets/segmentation_img.jpg" # THE EXAMPLE JPG/PNG IMAGE WHICH IS COPIED ON THE SERVER INITIALIZATION

ORIGINAL_IMAGE_PATH = "assets/uploaded_original_img" # THE ORIGINAL INITIAL IMAGE (EXCEPT FOR TIF IT IS THE SAME WITH THE DEFAULT_IMAGE_PATH)
DEFAULT_IMAGE_PATH = "assets/default_img"            # THE JPG/PNG IMAGE WITH THE INITIAL RESOLUTION
RESIZED_IMAGE_PATH = "assets/resized_img"            # THE JPG/PNG IMAGE WITH THE RESIZED RESOLUTION
PROCESS_IMAGE_PATH = "assets/process_img"            # THE IMAGE WHICH IS USED FOR THE CLASSIFIER (EXCEPT FOR TIF IT IS THE SAME WITH THE RESIZED_IMAGE_PATH)

os.remove(glob(ORIGINAL_IMAGE_PATH+'*')[0]) if glob(ORIGINAL_IMAGE_PATH+'*') else None
os.remove(glob(DEFAULT_IMAGE_PATH+'*')[0]) if glob(DEFAULT_IMAGE_PATH+'*') else None
os.remove(glob(RESIZED_IMAGE_PATH+'*')[0]) if glob(RESIZED_IMAGE_PATH+'*') else None
os.remove(glob(PROCESS_IMAGE_PATH+'*')[0]) if glob(PROCESS_IMAGE_PATH+'*') else None
shutil.copy(EXAMPLE_IMAGE_PATH, RESIZED_IMAGE_PATH + '.jpg')
shutil.copy(EXAMPLE_IMAGE_PATH, PROCESS_IMAGE_PATH + '.jpg')

SEG_FEATURE_TYPES = [" Ένταση", " Ακμές", " Υφή"]

SEG_FEATURE_TYPES_greek = {
    " Ένταση":"intensity",
    " Ακμές":"edges",
    " Υφή":"texture",
}

# the number of different classes for labels
NUM_LABEL_CLASSES = 5
DEFAULT_LABEL_CLASS = 0
class_label_colormap = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#FF00FB", "#B55400", "#1F1B4D", "#D6292C", "#40FF00"]
class_labels = list(range(NUM_LABEL_CLASSES))
# we can't have less colors than classes
assert NUM_LABEL_CLASSES <= len(class_label_colormap)

# Font and background colors associated with each theme
text_color = {"dark": "#95969A", "light": "#595959"}
card_color = {"dark": "#2D3038", "light": "#FFFFFF"}


def class_to_color(n):
    return class_label_colormap[n]


def color_to_class(c):
    return class_label_colormap.index(c)


img = skio.imread(glob(RESIZED_IMAGE_PATH + '*')[0])
img_pro = skio.imread(glob(PROCESS_IMAGE_PATH + '*')[0])
H_init, W_init, _ = img.shape

features_dict = {}

external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/segmentation-style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
app.title = "SOSAME Αρχαιολογική Στρωματογραφία"

auth = dash_auth.BasicAuth(
     app,
     VALID_USERNAME_PASSWORD_PAIRS
 )

def make_default_figure(
    images=[glob(RESIZED_IMAGE_PATH + '*')[0]],
    stroke_color=class_to_color(DEFAULT_LABEL_CLASS),
    stroke_width=DEFAULT_STROKE_WIDTH,
    shapes=[],
):
    fig = plot_common.dummy_fig()
    plot_common.add_layout_images_to_fig(fig, images)
    fig.update_layout(
        {
            "dragmode": "drawopenpath",
            "shapes": shapes,
            "newshape.line.color": stroke_color,
            "newshape.line.width": stroke_width,
            "margin": dict(l=0, r=0, b=0, t=0, pad=4),
        }
    )
    return fig


def shapes_to_key(shapes):
    return json.dumps(shapes)


def store_shapes_seg_pair(d, key, seg, remove_old=True):
    """
    Stores shapes and segmentation pair in dict d
    seg is a PIL.Image object
    if remove_old True, deletes all the old keys and values.
    """
    bytes_to_encode = io.BytesIO()
    seg.save(bytes_to_encode, format="png")
    bytes_to_encode.seek(0)
    data = base64.b64encode(bytes_to_encode.read()).decode()
    if remove_old:
        return {key: data}
    d[key] = data
    return d


def look_up_seg(d, key):
    """ Returns a PIL.Image object """
    data = d[key]
    img_bytes = base64.b64decode(data)
    img = PIL.Image.open(io.BytesIO(img_bytes))
    return img

# Header
header = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(
                            id="logo",
                            src=app.get_asset_url("sosame_logo.png"),
                            height="90px",
                        ),
                        md="auto",
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    #html.H2("SOSAME",
                                    #style={'color': '#5065a7'}),
                                    html.H4("Αρχαιολογική Στρωματογραφία",
                                    style={'color': '#5065a7'}),
                                ],
                                id="app-title"
                                
                            )
                        ],
                        md=True,
                        align="center"
                    ),
                ],
                align="center"
            ),
        ],
        fluid=True,
    ),
    dark=True,
    color="#f2eee3",
    sticky="top",
)

# # Description
# description = dbc.Col(
#     [
#         dbc.Card(
#             id="description-card",
#             children=[
#                 dbc.CardHeader("Επεξήγηση"),
#                 dbc.CardBody(
#                     [
#                         dbc.Row(
#                             [
#                                 dbc.Col(
#                                     [
#                                         html.Img(
#                                             src="assets/segmentation_img_example_marks.jpg",
#                                             width="200px",
#                                         )
#                                     ],
#                                     md="auto",
#                                 ),
#                                 dbc.Col(
#                                     html.P(
#                                         "Παράδειγμα ημι-αυτόματης ταξινόμησης αρχαιολογικής στρωματογραφίας. "
#                                         "Ο χρήστης χρησιμοποιώντας διαφορετικά χρώματα, πρέπει να επισημάνει "
#                                         'στην εικόνα τα διαφορετικά στρώματα εδάφους, όπως στο διπλανό παράδειγμα. '
#                                         "Στη συνέχεια, ενεργοποιήστε τον 'Υπολογισμός Κατάτμησης Εδάφους' για να δείτε την "
#                                         "ταξινόμηση με βάση την ψηφιοποίησή σας. Μπορείτε να προσθέσετε περισσότερες επισημάνσεις "
#                                         "για να διευκρινίσετε τμήματα της εικόνας όπου ο ταξινομητής δεν ήταν επιτυχής, ενημερώνοντας την ταξινόμηση."
#                                     ),
#                                     md=True,
#                                 ),
#                             ]
#                         ),
#                     ]
#                 ),
#             ],
#         )
#     ],
#     md=12,
# )

# Image Segmentation
segmentation = [
    dbc.Card(
        id="segmentation-card",
        children=[
            dbc.CardHeader(dcc.Markdown('''
                            ##### Βήματα
                            **1. Επιλέξτε εικόνα &ensp;** 
                            **2. Επιλέξτε αριθμό κατηγοριών &ensp;**
                            **3. Επισημάνετε τα διαφορετικά στρώματα &ensp;**
                            **4. Επιλέξτε "Υπολογισμός Κατάτμησης Εδάφους"**
                            '''),style={"font-size": "100%"}),
            dbc.CardBody(
                [dbc.Row(
                    # Wrap dcc.Loading in a div to force transparency when loading
                    dbc.Col(
                    html.Div(
                        id="transparent-loader-wrapper",
                        children=[
                            dcc.Loading(
                                id="segmentations-loading",
                                type="circle",
                                children=[
                                    # Graph
                                    dcc.Graph(
                                        id="graph",
                                        figure=make_default_figure(),
                                        config={
                                            "modeBarButtonsToAdd": [
                                                "drawrect",
                                                "drawopenpath",
                                                "eraseshape",
                                            ],"displaylogo": False
                                        },
                                    ),
                                ],
                            )
                        ]))),
                    dbc.Row([
                    # Wrap dcc.Loading in a div to force transparency when loading
                    dbc.Col(
                    html.Div([
                    html.H6("Αρχική εικόνα", style={'textAlign': 'center'}),
                    html.Div(
                        id="transparent-loader-wrapper-2",
                        children=[
                            dcc.Loading(
                                id="segmentations-loading-2",
                                type="circle",
                                children=[
                                    # Graph
                                    dcc.Graph(
                                        id="graph-2",
                                        figure=make_default_figure().update_layout(height = 255),
                                        config={"displayModeBar": False, 'staticPlot': True}
                                    ),
                                ],
                            )
                        ])])),
                    # Wrap dcc.Loading in a div to force transparency when loading
                    dbc.Col(
                    html.Div([
                    html.H6("Όρια στρωμάτων", style={'textAlign': 'center'}),
                    html.Div(
                        id="transparent-loader-wrapper-3",
                        children=[
                            dcc.Loading(
                                id="segmentations-loading-3",
                                type="circle",
                                children=[
                                    # Graph
                                    dcc.Graph(
                                        id="graph-3",
                                        figure=make_default_figure().update_layout(height = 255),
                                        config={"displayModeBar": False, 'staticPlot': True}
                                    ),
                                ],
                            )
                        ])]))
                     ], style={"margin-top":"20px"})],
            ),
            dbc.CardFooter(
                [html.Div([ 
                        dcc.Upload(
                            id='upload-image',
                            children=html.Div([
                                'Επιλέξτε εικόνα ',
                                html.A('(ή σύρετε και τοποθετήστε την εδώ)')
                            ]),
                            style={
                                "width": "100%",
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'textAlign': 'center'
                                
                                },
                            # Allow multiple files to be uploaded
                            multiple=False
                        ),
                    ], className="mb-3"),
                    
                    # Download links
                    html.A(id="download", download="classifier.json",),
                    html.Div(
                        children=[
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        "Κατεβάστε την ταξινομημένη εικόνα",
                                        id="download-image-button",
                                        outline=True,
                                    ),
                                    dbc.Button(
                                        "Κατεβάστε τα όρια στρωμάτων",
                                        id="download-image-button-boundaries",
                                        outline=True,
                                    ),
                                    dbc.Button(
                                        "Κατεβάστε την εικόνα κατάτμησης",
                                        id="download-image-button-int8",
                                        outline=True,
                                    ),
                                    dbc.Button(
                                        "Κατεβάστε τον ταξινομητή",
                                        id="download-button",
                                        outline=True,
                                    ),
                                ],
                                
                                style={
                                    "width": "100%",
                                    "height": "60px",
                                    'lineHeight': '60px',
                                    'borderWidth': '1px'
                                    
                                    },
                            ),
                        ],
                    ),
                    html.A(id="download-image", download="classified-image.png",),
                    html.A(id="download-image-boundaries", download="classified-image-boundaries.png",),
                    html.A(id="download-image-int8", download="classified-image-int8.png",),
                ]
            ),
        ],
    )
]

items = [dbc.DropdownMenuItem(str(i)) for i in range(1,10+1)]

# sidebar
sidebar = [
    dbc.Card(
        id="sidebar-card",
        children=[
            dbc.CardHeader("Εργαλεία"),
            dbc.CardBody(
                [
                    html.H6("Κατηγορίες Ταξινόμησης", className="card-title"),
                    html.Div([
                            dcc.Dropdown(
                            options = [{'label': str(i), 'value': i} for i in range(1,10+1)], value=NUM_LABEL_CLASSES , id="label-class-menu", placeholder="How many labels"),
                            html.Div(id='label-class-menu-container')],
                            className="mb-3",
                            ),
                    
                    # Label class chosen with buttons
                    html.Div(
                        id="label-class-buttons",
                        children=[
                            dbc.Button(
                                "%2d" % (n+1,),
                                id={"type": "label-class-button", "index": n},
                                style={"background-color": class_to_color(c)},
                            )
                            for n, c in enumerate(class_labels)
                        ],
                    ),
                    html.Hr(),
                    dbc.Form(
                        [
                            dbc.FormGroup(
                                [
                                    dbc.Label(
                                        "Πλάτος εργαλείου ψηφιοποίησης",
                                        html_for="stroke-width",
                                    ),
                                    # Slider for specifying stroke width
                                    dcc.Slider(
                                        id="stroke-width",
                                        min=0,
                                        max=6,
                                        step=0.1,
                                        value=DEFAULT_STROKE_WIDTH,
                                    ),
                                ]
                            ),
                            dbc.FormGroup(
                                [
                                    html.H6(
                                        id="stroke-width-display",
                                        className="card-title",
                                    ),
                                    dbc.Label(
                                        "Εύρος παραμέτρου θαμπώματος",
                                        html_for="sigma-range-slider",
                                    ),
                                    dcc.RangeSlider(
                                        id="sigma-range-slider",
                                        min=0.01,
                                        max=20,
                                        step=0.01,
                                        value=[0.5, 16],
                                    ),
                                ]
                            ),
                            dbc.FormGroup(
                                [
                                    dbc.Label(
                                        "Επιλογή Χαρακτηριστικών για Ταξινόμηση",
                                        html_for="segmentation-features",
                                    ),
                                    dcc.Checklist(
                                        id="segmentation-features",
                                        options=[
                                            {"label": l, "value": l}
                                            for l in SEG_FEATURE_TYPES
                                        ],
                                        value=SEG_FEATURE_TYPES,
                                    ),
                                ]
                            ),
                            # Indicate showing most recently computed segmentation
                            dcc.Checklist(
                                id="show-segmentation",
                                options=[
                                    {
                                        "label": " Υπολογισμός Κατάτμησης Εδάφους",
                                        "value": "Show segmentation",
                                    }
                                ],
                                value=[], style={"font-size": "120%", "font-weight": "bold"}
                            ),
                        ]
                    ),
                ]
            ),
        ],
    ),
]

meta = [
    html.Div(
        id="no-display",
        children=[
            # Store for user created masks
            # data is a list of dicts describing shapes
            dcc.Store(id="masks", data={"shapes": []}),
            dcc.Store(id="classifier-store", data={}),
            dcc.Store(id="classified-image-store", data=""),
            dcc.Store(id="classified-image-store-int8", data=""),
            dcc.Store(id="features_hash", data=""),
        ],
    ),
    html.Div(id="download-dummy"),
    html.Div(id="download-image-dummy"),
    html.Div(id="download-image-dummy-int8"),
]

footer = html.Footer(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(
                            id="logo-2",
                            src=app.get_asset_url("sosame_logo_2.png"),
                            height="90px",
                        ),
                        md="auto", align="center", width={"offset": 3}
                    ),
                    dbc.Col(
                        html.Img(
                            id="logo-3",
                            src=app.get_asset_url("espa_logo.png"),
                            height="90px",
                        ),
                        md="auto", align="center", width={"offset": 2}
                    ),
                ], align="center", style={"margin-top":"50px"}
            ),
        ],
        fluid=True,
    )
)

app.layout = html.Div(
    [
        header,
        dbc.Container(
            [
                #dbc.Row(description),
                dbc.Row(
                    id="app-content",
                    children=[dbc.Col(segmentation, md=8), dbc.Col(sidebar, md=4)],
                ),
                dbc.Row(dbc.Col(meta)),
            ],
            fluid=True,
        ),
        footer,
    dcc.Location(id='url', refresh=True),]
)


# Converts image classifier to a JSON compatible encoding and creates a
# dictionary that can be downloaded
# see use_ml_image_segmentation_classifier.py
def save_img_classifier(clf, label_to_colors_args, segmenter_args):
    clfbytes = io.BytesIO()
    pickle.dump(clf, clfbytes)
    clfb64 = base64.b64encode(clfbytes.getvalue()).decode()
    return {
        "classifier": clfb64,
        "segmenter_args": segmenter_args,
        "label_to_colors_args": label_to_colors_args,
    }


def show_segmentation(image_path, mask_shapes, features, segmenter_args):
    """ adds an image showing segmentations to a figure's layout """
    # add 1 because classifier takes 0 to mean no mask
    shape_layers = [color_to_class(shape["line"]["color"]) + 1 for shape in mask_shapes]
    label_to_colors_args = {
        "colormap": class_label_colormap,
        "color_class_offset": -1,
    }
    boundimg, segimg, seg, clf = compute_segmentations(
        mask_shapes,
        img_path=image_path,
        shape_layers=shape_layers,
        label_to_colors_args=label_to_colors_args,
        features=features,
    )
    # get the classifier that we can later store in the Store
    classifier = save_img_classifier(clf, label_to_colors_args, segmenter_args)
    segimgpng = plot_common.img_array_to_pil_image(segimg)
    boundimgpng = plot_common.img_array_to_pil_image(boundimg)
    segpng = plot_common.img_array_to_pil_image(seg)
    return (segimgpng, boundimgpng, segpng, classifier)

@app.callback(
    Output('label-class-menu-container', 'children'),
    [Input('label-class-menu', 'value')]
)
def update_output(value):
    return f'{value} επιλεγμένες κατηγορίες'


@app.callback(
    Output('label-class-buttons', 'children'),
    [Input('label-class-menu', 'value')]
)
def update_div_buttons(value):
    class_labels = list(range(value))
    return [dbc.Button(
                                "%2d" % (n+1,),
                                id={"type": "label-class-button", "index": n},
                                style={"background-color": class_to_color(c)},
                            )
                            for n, c in enumerate(class_labels)
                        ]

@app.callback(Output("url", "href"),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
              State('upload-image', 'last_modified')])
def upload_image_file(content, file_name, timestamp):
    if content is not None:
        global img
        global img_pro
        global H_init
        global W_init
        
        _, f_extension = os.path.splitext(file_name)
        f_extension = f_extension.lower()
        
        for f in glob(DEFAULT_IMAGE_PATH+"*") + glob(RESIZED_IMAGE_PATH+"*"):
            os.remove(f)
        
        ORIGINAL_IMAGE_PATH_local = ORIGINAL_IMAGE_PATH + f_extension
        PROCESS_IMAGE_PATH_local = PROCESS_IMAGE_PATH + f_extension
        
        content_type, content_string = content.split(',')
        imgdata = base64.b64decode(content_string)
        
        with open(ORIGINAL_IMAGE_PATH_local, 'wb') as f:
            f.write(imgdata)

        if '.tif' in f_extension:
            
            DEFAULT_IMAGE_PATH_local = DEFAULT_IMAGE_PATH + '.png'
            RESIZED_IMAGE_PATH_local = RESIZED_IMAGE_PATH + '.png'
            
            with rasterio.open(ORIGINAL_IMAGE_PATH_local, 'r') as f:
                img_orig = f.read() # [GRE, RED, REG, NIR] (4, H, W)
                
                img_process = np.moveaxis(img_orig, [-2, -1], [0, 1])
                tags = f.tags().copy()
                meta = f.meta
                
                img_orig = img_process[:,:,[1,3,2]] 
                
                min_percent = 4   # Low percentile
                max_percent = 96  # High percentile
                
                img_init = np.zeros(shape = img_orig.shape, dtype = np.uint8)
                
                for band in range(img_orig.shape[2]):
                    img_band = img_orig[:,:,band]
                    
                    lo, hi = np.percentile(img_band, (min_percent, max_percent))
                    
                    # Apply linear "stretch" - lo goes to 0, and hi goes to 1
                    res_img = (img_band.astype(float) - lo) / (hi-lo)
                    
                    #Multiply by 255, clamp range to [0, 255] and convert to uint8
                    res_img = np.maximum(np.minimum(res_img*255, 255), 0).astype(np.uint8)
                    
                    img_init[:,:,band] = res_img
                
                skio.imsave(DEFAULT_IMAGE_PATH_local, img_init)
                
        else:
            DEFAULT_IMAGE_PATH_local = DEFAULT_IMAGE_PATH + f_extension
            RESIZED_IMAGE_PATH_local = RESIZED_IMAGE_PATH + f_extension
            
            with open(DEFAULT_IMAGE_PATH_local, 'wb') as f:
                f.write(imgdata)
            img_init = skio.imread(DEFAULT_IMAGE_PATH_local)
            img_process = img_init
        
        print('Uploaded image '+ file_name + ' at:' + RESIZED_IMAGE_PATH_local)
        
        # Resize Image if it is too big
        H_init, W_init, _ = img_init.shape
        img_init_type = img_init.dtype
        img_process_type = img_process.dtype
    
        if W_init > 1200 or H_init >1200:
    
            ratio = H_init / W_init
            if H_init>=W_init:
    
                H_new = 1200
                W_new = int(H_new / ratio)
            else:
    
                W_new = 1200
                H_new = int(W_new * ratio)
    
            img = resize(img_init,  (H_new, W_new), preserve_range = True, anti_aliasing = False).astype(img_init_type)
            img_pro = resize(img_process,  (H_new, W_new), preserve_range = True, anti_aliasing = False).astype(img_process_type)
        else:
            img = img_init
            img_pro = img_process
            
        
        skio.imsave(RESIZED_IMAGE_PATH_local, img)
        
        if '.tif' in f_extension:
            if W_init > 1200 or H_init >1200:
                meta['height'] = H_new
                meta['width'] = W_new

            with rasterio.open(PROCESS_IMAGE_PATH_local, 'w', **meta) as f:
                f.write(np.moveaxis(img_pro, [0, 1], [-2, -1]))
                f.update_tags(**tags)
        else:
            skio.imsave(PROCESS_IMAGE_PATH_local, img_pro)
        
        return "/"

@app.callback(
    [
        Output("graph", "figure"),
        Output("graph-2", "figure"),
        Output("graph-3", "figure"),
        Output("masks", "data"),
        Output("stroke-width-display", "children"),
        Output("classifier-store", "data"),
        Output("classified-image-store", "data"),
        Output("classified-image-store-int8", "data"),
    ],
    [
        Input("graph", "relayoutData"),
        Input(
            {"type": "label-class-button", "index": dash.dependencies.ALL},
            "n_clicks_timestamp",
        ),
        Input("stroke-width", "value"),
        Input("show-segmentation", "value"),
        Input("download-button", "n_clicks"),
        Input("download-image-button", "n_clicks"),
        Input("download-image-button-boundaries", "n_clicks"),
        Input("download-image-button-int8", "n_clicks"),
        Input("segmentation-features", "value"),
        Input("sigma-range-slider", "value")],
    [State("masks", "data"),],
)
def annotation_react(
    graph_relayoutData,
    any_label_class_button_value,
    stroke_width_value,
    show_segmentation_value,
    download_button_n_clicks,
    download_image_button_n_clicks,
    download_image_button_n_clicks_boundaries,
    download_image_button_n_clicks_int8,
    segmentation_features_value,
    sigma_range_slider_value,
    masks_data,
):

    classified_image_store_data = dash.no_update
    classifier_store_data = dash.no_update
    classified_image_store_data_int8 = dash.no_update
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    if cbcontext in ["segmentation-features.value", "sigma-range-slider.value"] or (
        ("Show segmentation" in show_segmentation_value)
        and (len(masks_data["shapes"]) > 0)
    ):
        segmentation_features_dict = {
            "intensity": False,
            "edges": False,
            "texture": False,
        }
        
        for feat in segmentation_features_value:
            segmentation_features_dict[SEG_FEATURE_TYPES_greek[feat]] = True
        t1 = time()
        print(img_pro.shape)
        features = compute_features(
            img_pro,
            **segmentation_features_dict,
            sigma_min=sigma_range_slider_value[0],
            sigma_max=sigma_range_slider_value[1],
        )
        print(features.shape)
        t2 = time()
        print(t2 - t1)
    if cbcontext == "graph.relayoutData":
        if "shapes" in graph_relayoutData.keys():
            masks_data["shapes"] = graph_relayoutData["shapes"]
        else:
            return dash.no_update
    stroke_width = int(round(2 ** (stroke_width_value)))
    # find label class value by finding button with the most recent click
    if any_label_class_button_value is None:
        label_class_value = DEFAULT_LABEL_CLASS
    else:
        label_class_value = max(
            enumerate(any_label_class_button_value),
            key=lambda t: 0 if t[1] is None else t[1],
        )[0]

    fig = make_default_figure(
        images=[glob(RESIZED_IMAGE_PATH + '*')[0]],
        stroke_color=class_to_color(label_class_value),
        stroke_width=stroke_width,
        shapes=masks_data["shapes"],
    )
    fig_2 = make_default_figure(images=[glob(RESIZED_IMAGE_PATH + '*')[0]])
    fig_3 = make_default_figure(images=[glob(RESIZED_IMAGE_PATH + '*')[0]])
    
    # We want the segmentation to be computed
    if ("Show segmentation" in show_segmentation_value) and (
        len(masks_data["shapes"]) > 0
    ):
        segimgpng = None
        try:
            feature_opts = dict(segmentation_features_dict=segmentation_features_dict)
            feature_opts["sigma_min"] = sigma_range_slider_value[0]
            feature_opts["sigma_max"] = sigma_range_slider_value[1]
            segimgpng, boundaryimgpng, segpng, clf = show_segmentation(
                glob(RESIZED_IMAGE_PATH + '*')[0], masks_data["shapes"], features, feature_opts
            )
            if cbcontext == "download-button.n_clicks":
                classifier_store_data = clf
            if cbcontext == "download-image-button.n_clicks":
                if (W_init, H_init) != segpng.size:
                    classified_image_store_data = plot_common.pil_image_to_uri(
                        blend_image_and_classified_regions_pil(
                            ImageOps.exif_transpose(PIL.Image.open(glob(DEFAULT_IMAGE_PATH + '*')[0])), segimgpng.resize((W_init, H_init), PIL.Image.NEAREST)
                        )
                    )
                else:
                    classified_image_store_data = plot_common.pil_image_to_uri(
                        blend_image_and_classified_regions_pil(
                            PIL.Image.open(glob(RESIZED_IMAGE_PATH + '*')[0]), segimgpng
                        )
                    )
            if cbcontext == "download-image-button-boundaries.n_clicks":
                if (W_init, H_init) != boundaryimgpng.size:
                    classified_image_store_data = plot_common.pil_image_to_uri(
                        blend_image_and_classified_regions_pil(
                            ImageOps.exif_transpose(PIL.Image.open(glob(DEFAULT_IMAGE_PATH + '*')[0])), boundaryimgpng.resize((W_init, H_init), PIL.Image.NEAREST)
                        )
                    )
                else:
                    classified_image_store_data = plot_common.pil_image_to_uri(
                        blend_image_and_classified_regions_pil(
                            PIL.Image.open(glob(RESIZED_IMAGE_PATH + '*')[0]), boundaryimgpng
                        )
                    )
            if cbcontext == "download-image-button-int8.n_clicks":
                if (W_init, H_init) != segpng.size:
                    classified_image_store_data_int8 = plot_common.pil_image_to_uri(segpng.resize((W_init, H_init), PIL.Image.NEAREST))
                else:
                    classified_image_store_data_int8 = plot_common.pil_image_to_uri(segpng)
        except ValueError:
            # if segmentation fails, draw nothing
            pass
        images_to_draw = []
        if segimgpng is not None:
            images_to_draw = [segimgpng]
        if boundaryimgpng is not None:
            bd_images_to_draw = [boundaryimgpng]
        fig = plot_common.add_layout_images_to_fig(fig, images_to_draw)
        fig_3 = plot_common.add_layout_images_to_fig(fig_3, bd_images_to_draw)
    fig.update_layout(uirevision="segmentation")
    fig_2.update_layout(uirevision="segmentation-2")
    fig_3.update_layout(uirevision="segmentation-2")
    return (
        fig,
        fig_2,
        fig_3,
        masks_data,
        "Επιλεγμένο πλάτος: %d" % (stroke_width,),
        classifier_store_data,
        classified_image_store_data,
        classified_image_store_data_int8,
    )


# set the download url to the contents of the classifier-store (so they can be
# downloaded from the browser's memory)
app.clientside_callback(
    """
function(the_store_data) {
    let s = JSON.stringify(the_store_data);
    let b = new Blob([s],{type: 'text/plain'});
    let url = URL.createObjectURL(b);
    return url;
}
""",
    Output("download", "href"),
    [Input("classifier-store", "data")],
)


# set the download url to the contents of the classified-image-store (so they can be
# downloaded from the browser's memory)
app.clientside_callback(
    """
function(the_image_store_data) {
    return the_image_store_data;
}
""",
    Output("download-image", "href"),
    [Input("classified-image-store", "data")],
)

# set the download url to the contents of the classified-image-store (so they can be
# downloaded from the browser's memory)
app.clientside_callback(
    """
function(the_image_store_data) {
    return the_image_store_data;
}
""",
    Output("download-image-boundaries", "href"),
    [Input("classified-image-store-boundaries", "data")],
)

# set the download url to the contents of the classified-image-store (so they can be
# downloaded from the browser's memory)
app.clientside_callback(
    """
function(the_image_store_data) {
    return the_image_store_data;
}
""",
    Output("download-image-int8", "href"),
    [Input("classified-image-store-int8", "data")],
)

# simulate a click on the <a> element when download.href is updated
app.clientside_callback(
    """
function (download_href) {
    let elem = document.querySelector('#download');
    elem.click()
    return "";
}
""",
    Output("download-dummy", "children"),
    [Input("download", "href")],
)

# simulate a click on the <a> element when download.href is updated
app.clientside_callback(
    """
function (download_image_href) {
    let elem = document.querySelector('#download-image');
    elem.click()
    return "";
}
""",
    Output("download-image-dummy", "children"),
    [Input("download-image", "href")],
)

# simulate a click on the <a> element when download.href is updated
app.clientside_callback(
    """
function (download_image_href) {
    let elem = document.querySelector('#download-image-boundaries');
    elem.click()
    return "";
}
""",
    Output("download-image-dummy-boundaries", "children"),
    [Input("download-image-boundaries", "href")],
)

# simulate a click on the <a> element when download.href is updated
app.clientside_callback(
    """
function (download_image_href) {
    let elem = document.querySelector('#download-image-int8');
    elem.click()
    return "";
}
""",
    Output("download-image-dummy-int8", "children"),
    [Input("download-image-int8", "href")],
)


if __name__ == "__main__":
    app.run_server()