import requests


def detect_emotion(text):
    """Sends a request to the emotion detection API and returns the detected emotions.

        Args:
            text (str): The text input for which the emotion needs to be detected.

        Returns:
            dict: A dictionary containing the detected emotions (anger, disgust, fear, joy, sadness)
                  and the dominant emotion. The format is:
                  {
                      'anger': float or None,
                      'disgust': float or None,
                      'fear': float or None,
                      'joy': float or None,
                      'sadness': float or None,
                      'dominant_emotion': str or None
                  }
    """

    url = ('https://sn-watson-emotion.labs.skills.network/v1'
           '/watson.runtime.nlp.v1/NlpService/EmotionPredict')

    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }

    payload = {
        "raw_document": {
            "text": text
        }
    }

    response = requests.post(url, json=payload, headers=headers, timeout=5)  # Added timeout

    data = {
        'anger': None,
        'disgust': None,
        'fear': None,
        'joy': None,
        'sadness': None,
        'dominant_emotion': None
    }

    if response.status_code == 200:
        response_data = response.json()
        data = response_data['emotionPredictions'][0]['emotion']
        dominant_emotion = max(data, key=data.get)
        data['dominant_emotion'] = dominant_emotion

    return data


def predict_emotion(emotion_data):
    """Processes emotion data to extract relevant information.

    Args:
        emotion_data (dict): The emotion data returned from the detect_emotion function.
                             Expected format:
                             {
                                 'emotionPredictions': [
                                     {
                                         'emotion': {
                                             'anger': float or None,
                                             'disgust': float or None,
                                             'fear': float or None,
                                             'joy': float or None,
                                             'sadness': float or None
                                         }
                                     }
                                 ]
                             }

    Returns:
        dict: A dictionary containing the processed emotions and the dominant emotion.
              The format is:
              {
                  'anger': float or None,
                  'disgust': float or None,
                  'fear': float or None,
                  'joy': float or None,
                  'sadness': float or None,
                  'dominant_emotion': str or None
              }
    """

    if all(value is None for value in emotion_data.values()):
        return emotion_data

    if 'emotionPredictions' in emotion_data and emotion_data['emotionPredictions']:
        emotions = emotion_data['emotionPredictions'][0]['emotion']
        dominant_emotion = max(emotions, key=emotions.get)

        return {
            'anger': emotions.get('anger'),
            'disgust': emotions.get('disgust'),
            'fear': emotions.get('fear'),
            'joy': emotions.get('joy'),
            'sadness': emotions.get('sadness'),
            'dominant_emotion': dominant_emotion
        }

    return {
        'anger': None,
        'disgust': None,
        'fear': None,
        'joy': None,
        'sadness': None,
        'dominant_emotion': None
    }
