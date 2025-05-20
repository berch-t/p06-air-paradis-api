import unittest
import json
from api import app, preprocess_tweet

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_check(self):
        response = self.app.get('/health')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('status', data)
        # Le statut peut être "healthy" ou "degraded" selon que le modèle est chargé ou non
        self.assertIn(data['status'], ['healthy', 'degraded'])

    def test_preprocess_tweet(self):
        # Test du prétraitement de tweet
        tweet = "@user This is a #test tweet with http://example.com URL and some numbers 123!"
        processed = preprocess_tweet(tweet)
        # Vérification de la suppression des mentions, hashtags, URLs, etc.
        self.assertNotIn('@user', processed)
        self.assertNotIn('http', processed)
        self.assertNotIn('123', processed)
        self.assertIn('test', processed)  # Le mot 'test' devrait être conservé
        
if __name__ == '__main__':
    unittest.main()