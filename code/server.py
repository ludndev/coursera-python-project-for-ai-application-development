import sys
from flask import Flask, render_template, request, jsonify
from EmotionDetection.emotion_detection import detect_emotion, predict_emotion  # Specific imports

app = Flask(__name__, static_folder='static', template_folder='templates', static_url_path='/')


@app.route("/emotionDetector")
def sent_detector():
    """Detects the emotion of the provided text and returns the results."""

    text_to_detect = request.args.get('textToAnalyze')

    if not text_to_detect:
        return jsonify({'error': 'Invalid request. '
                                 'Please provide missing query textToAnalyze parameter.'}), 400

    response = detect_emotion(text_to_detect)
    formatted_response = predict_emotion(response)

    if formatted_response['dominant_emotion'] is None:
        return jsonify({'error': 'Invalid text! Please try again.'}), 500

    return jsonify({
        'anger': formatted_response['anger'],
        'disgust': formatted_response['disgust'],
        'fear': formatted_response['fear'],
        'joy': formatted_response['joy'],
        'sadness': formatted_response['sadness'],
        'dominant_emotion': formatted_response['dominant_emotion']
    })


@app.route("/")
def render_index_page():
    """Renders the index page."""

    return render_template('index.html')


@app.errorhandler(500)
def internal_server_error(_):
    """Handles internal server errors."""

    return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == "__main__":
    """Runs the Flask web application."""
    try:
        app.run(host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down server")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        sys.exit(1)
