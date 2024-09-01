from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))      
        )
        predict_pipeline = PredictPipeline()

        df = data.get_data_as_data_frame()
        logging.info('dataframe: {df}')

        prediction = predict_pipeline.predict(df)
        logging.info('predection: {prediction}')

        return render_template('index.html', results= prediction[0])

if __name__ == '__main__':
    app.run()


