from flask import Flask, request,  render_template, send_file, redirect, url_for
from pathlib import Path
from src.pipeline.prediction_pipeline import PredictionFileValidation,PredictionPipeline
from src.utils import clear_directory

app = Flask(__name__)

prediction_data_filepath = Path("Prediction_data_raw")
prediction_result_filepath = Path("Prediction_result", "result.csv")
schema_path = Path("artifacts", "schema.json")

@app.route('/', methods=['GET', 'POST'])
@app.route('/upload', methods=['GET', 'POST'])
def upload_files():
    if request.method=="GET":
        return render_template("index.html", result=None, message=None)

    if "csvFiles" not in request.files:
        return redirect(request.url)
    
    try:
        prediction_data_filepath.mkdir(exist_ok=True)
        upload_files = request.files.getlist("csvFiles")

        for file in upload_files:
            file.save(Path(prediction_data_filepath, file.filename))

        if not any(prediction_data_filepath.glob("*.csv")):
            # checking for csv files in the uploaded files
            message = "Error: upload '.csv' files"
            return render_template("index.html", result=None, message=message, schema=url_for("schema"))

        file_validation = PredictionFileValidation()
        prediction_df = file_validation.validate(prediction_data_filepath)

        if len(prediction_df)==0:
            # checking if data exists in dataframe
            message = "Error: valid data not found inside the files.\nData must be compatible with the schema."
            return render_template("index.html", result=None, message=message, schema=url_for("schema"))
        
        predictor_obj = PredictionPipeline()
        result = predictor_obj.predict(prediction_df)
        prediction_result_filepath.parent.mkdir(exist_ok=True)
        result.to_csv(prediction_result_filepath, index=False,header=["Wafer ID","Result"])

        return render_template("index.html", result=url_for("result"), message=None)
        
    finally:
        # clearing the created directories
        clear_directory(prediction_data_filepath)

@app.route("/result")
def result():
    return send_file(prediction_result_filepath, as_attachment=True)

@app.route(f"/artifacts/schema.json")
def schema():
    return send_file(schema_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=7000)