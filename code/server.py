from flask import Flask, render_template, request, jsonify
from EmotionDetection.emotion_detection import *

app = Flask(__name__, static_folder='static', template_folder='templates', static_url_path='/')


@app.route("/emotionDetector")
def sent_detector():
    text_to_detect = request.args.get('textToAnalyze')

    if not text_to_detect:
        return jsonify({'error': 'Invalid request. Please provide missing query textToAnalyze parameter.'}), 400

    response = detect_emotion(text_to_detect)
    formated_response = predict_emotion(response)

    if formated_response['dominant_emotion'] is None:
        return jsonify({'error': 'Invalid text! Please try again.'}), 500

    return jsonify({
        'anger': formated_response['anger'],
        'disgust': formated_response['disgust'],
        'fear': formated_response['fear'],
        'joy': formated_response['joy'],
        'sadness': formated_response['sadness'],
        'dominant_emotion': formated_response['dominant_emotion']
    })


@app.route("/")
def render_index_page():
    return render_template('index.html')


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down server")
    except Exception as e:
        print(e)
    finally:
        quit(1)
