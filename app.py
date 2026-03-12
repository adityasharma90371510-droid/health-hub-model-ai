from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import pandas as pd
from datetime import datetime

from utils_predictor import predict_image
from gradcam import generate_gradcam

app = Flask(__name__)

app.static_folder = "static"

UPLOAD_FOLDER = "static/uploads"
LOG_FILE = "outputs/prediction_logs.xlsx"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

classes = [
"akiec",
"bcc",
"bkl",
"df",
"mel",
"nv",
"vasc"
]

disease_info = {

"mel":{
"name":"Melanoma",
"risk":"HIGH",
"description":"Melanoma is a dangerous skin cancer arising from melanocytes.",
"advice":"Consult a dermatologist immediately."
},

"nv":{
"name":"Melanocytic Nevus",
"risk":"LOW",
"description":"A benign mole composed of melanocytes.",
"advice":"Usually harmless but monitor changes."
},

"bcc":{
"name":"Basal Cell Carcinoma",
"risk":"MEDIUM",
"description":"A common skin cancer caused by sun exposure.",
"advice":"Consult dermatologist."
},

"akiec":{
"name":"Actinic Keratosis",
"risk":"MEDIUM",
"description":"Pre-cancerous lesion caused by sun damage.",
"advice":"Seek dermatological evaluation."
},

"bkl":{
"name":"Benign Keratosis",
"risk":"LOW",
"description":"Non-cancerous skin growth.",
"advice":"Usually harmless."
},

"df":{
"name":"Dermatofibroma",
"risk":"LOW",
"description":"Benign skin nodule.",
"advice":"Generally harmless."
},

"vasc":{
"name":"Vascular Lesion",
"risk":"LOW",
"description":"Blood vessel related lesion.",
"advice":"Usually benign."
}

}


def log_prediction(disease,confidence,risk):

    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_data=pd.DataFrame({
        "timestamp":[timestamp],
        "disease":[disease],
        "confidence":[confidence],
        "risk":[risk]
    })

    if os.path.exists(LOG_FILE):

        existing=pd.read_excel(LOG_FILE)

        updated=pd.concat([existing,new_data],ignore_index=True)

        updated.to_excel(LOG_FILE,index=False)

    else:

        new_data.to_excel(LOG_FILE,index=False)


@app.route("/", methods=["GET","POST"])
def index():

    prediction=None
    confidence=None
    image_path=None
    gradcam_path=None

    disease_name=None
    risk=None
    description=None
    advice=None

    chart_labels=[]
    chart_values=[]

    if request.method=="POST":

        file=request.files["image"]

        filename=secure_filename(file.filename)

        filepath=os.path.join(app.config["UPLOAD_FOLDER"],filename)

        file.save(filepath)

        prediction,confidence,probs=predict_image(filepath)

        gradcam_file=generate_gradcam(filepath)

        image_path="uploads/"+filename
        gradcam_path="uploads/"+gradcam_file

        info=disease_info.get(prediction)

        if info:

            disease_name=info["name"]
            risk=info["risk"]
            description=info["description"]
            advice=info["advice"]

        chart_labels=classes
        chart_values=[float(p*100) for p in probs]

        # LOG prediction to Excel
        log_prediction(disease_name,confidence,risk)

    return render_template(
    "index.html",
    prediction=disease_name,
    confidence=confidence,
    image_path=image_path,
    gradcam_path=gradcam_path,
    risk=risk,
    description=description,
    advice=advice,
    chart_labels=chart_labels,
    chart_values=chart_values,
    disease_chart="outputs/disease_distribution.png",
    risk_chart="outputs/risk_distribution.png",
    confidence_chart="outputs/confidence_distribution.png"
)


if __name__=="__main__":
    app.run(debug=True)