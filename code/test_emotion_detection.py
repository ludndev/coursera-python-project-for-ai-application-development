import unittest
from EmotionDetection.emotion_detection import *


class TestEmotionDetection(unittest.TestCase):
    def test_emotion_prediction(self):
        happy_detection = {
            'emotionPredictions': [
                {'emotion': {'anger': 0.1, 'disgust': 0.05, 'fear': 0.05, 'joy': 0.7, 'sadness': 0.1}}]
        }

        angry_detection = {
            'emotionPredictions': [
                {'emotion': {'anger': 0.7, 'disgust': 0.1, 'fear': 0.05, 'joy': 0.05, 'sadness': 0.1}}]
        }

        disgusted_detection = {
            'emotionPredictions': [
                {'emotion': {'anger': 0.05, 'disgust': 0.7, 'fear': 0.05, 'joy': 0.05, 'sadness': 0.1}}]
        }

        sad_detection = {
            'emotionPredictions': [
                {'emotion': {'anger': 0.1, 'disgust': 0.1, 'fear': 0.05, 'joy': 0.05, 'sadness': 0.7}}]
        }

        afraid_detection = {
            'emotionPredictions': [
                {'emotion': {'anger': 0.1, 'disgust': 0.05, 'fear': 0.7, 'joy': 0.05, 'sadness': 0.1}}]
        }

        # Run tests on each predefined emotion detection output
        self.assertEqual(predict_emotion(happy_detection)['dominant_emotion'], 'joy')
        self.assertEqual(predict_emotion(angry_detection)['dominant_emotion'], 'anger')
        self.assertEqual(predict_emotion(disgusted_detection)['dominant_emotion'], 'disgust')
        self.assertEqual(predict_emotion(sad_detection)['dominant_emotion'], 'sadness')
        self.assertEqual(predict_emotion(afraid_detection)['dominant_emotion'], 'fear')

    def test_detect_emotion(self):
        text_happy = "I am so thrilled to be here today!"
        text_angry = "This makes me so furious!"
        text_disgusted = "Just thinking about it is disgusting."
        text_sad = "I feel so down and depressed."
        text_afraid = "I'm really scared of what might happen."

        self.assertEqual(detect_emotion(text_happy)['dominant_emotion'], 'joy')
        self.assertEqual(detect_emotion(text_angry)['dominant_emotion'], 'anger')
        self.assertEqual(detect_emotion(text_disgusted)['dominant_emotion'], 'disgust')
        self.assertEqual(detect_emotion(text_sad)['dominant_emotion'], 'sadness')
        self.assertEqual(detect_emotion(text_afraid)['dominant_emotion'], 'fear')


if __name__ == '__main__':
    unittest.main()
