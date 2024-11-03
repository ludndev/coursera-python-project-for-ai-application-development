import requests
import json


def detect_emotion(text):
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'

    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }

    payload = {
        "raw_document": {
            "text": text
        }
    }

    response = requests.post(url, json=payload, headers=headers)

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
