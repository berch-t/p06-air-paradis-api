import unittest
import json
import sys
import os
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au chemin Python pour pouvoir importer l'API
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests.api.api as api_backup

class TestSentimentAPI(unittest.TestCase):
    """Tests unitaires pour l'API d'analyse de sentiment"""

    def setUp(self):
        """Configuration initiale pour les tests"""
        self.app = api_backup.app.test_client()
        self.app.testing = True
        
    def test_health_check(self):
        """Test de l'endpoint de vérification de l'état de l'API"""
        response = self.app.get('/health')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['model'], 'sentiment_analysis')
        
    @patch('api.preprocess_tweet')
    @patch('api.model.predict')
    @patch('api.tokenizer.texts_to_sequences')
    @patch('api.pad_sequences')
    def test_predict_positive_sentiment(self, mock_pad_sequences, mock_texts_to_sequences, 
                                          mock_predict, mock_preprocess):
        """Test de prédiction d'un sentiment positif"""
        # Configuration des mocks
        mock_preprocess.return_value = "processed tweet"
        mock_texts_to_sequences.return_value = [[1, 2, 3]]
        mock_pad_sequences.return_value = [[1, 2, 3, 0, 0]]
        mock_predict.return_value = [[0.8]]  # 0.8 > 0.5, donc sentiment positif
        
        # Envoi de la requête
        response = self.app.post(
            '/predict',
            data=json.dumps({'tweet': 'This is a great service!'}),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        # Vérification des résultats
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['sentiment'], 'Positif')
        self.assertGreater(data['confidence'], 0.7)
        
        # Vérification que les mocks ont été appelés correctement
        mock_preprocess.assert_called_once_with('This is a great service!')
        mock_texts_to_sequences.assert_called_once()
        mock_pad_sequences.assert_called_once()
        mock_predict.assert_called_once()
        
    @patch('api.preprocess_tweet')
    @patch('api.model.predict')
    @patch('api.tokenizer.texts_to_sequences')
    @patch('api.pad_sequences')
    def test_predict_negative_sentiment(self, mock_pad_sequences, mock_texts_to_sequences, 
                                          mock_predict, mock_preprocess):
        """Test de prédiction d'un sentiment négatif"""
        # Configuration des mocks
        mock_preprocess.return_value = "processed tweet"
        mock_texts_to_sequences.return_value = [[1, 2, 3]]
        mock_pad_sequences.return_value = [[1, 2, 3, 0, 0]]
        mock_predict.return_value = [[0.2]]  # 0.2 < 0.5, donc sentiment négatif
        
        # Envoi de la requête
        response = self.app.post(
            '/predict',
            data=json.dumps({'tweet': 'This service is terrible!'}),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        # Vérification des résultats
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['sentiment'], 'Négatif')
        self.assertGreater(data['confidence'], 0.7)
        
    def test_predict_empty_tweet(self):
        """Test avec un tweet vide"""
        response = self.app.post(
            '/predict',
            data=json.dumps({'tweet': ''}),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
        
    @patch('api.send_to_appinsights')
    def test_feedback_correct_prediction(self, mock_send_to_appinsights):
        """Test de l'endpoint de feedback pour une prédiction correcte"""
        response = self.app.post(
            '/feedback',
            data=json.dumps({
                'tweet': 'This is a great service!',
                'predicted_sentiment': 'Positif',
                'is_correct': True
            }),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        
        # Le feedback positif ne devrait pas envoyer de trace à Application Insights
        mock_send_to_appinsights.assert_not_called()
        
    @patch('api.send_to_appinsights')
    def test_feedback_incorrect_prediction(self, mock_send_to_appinsights):
        """Test de l'endpoint de feedback pour une prédiction incorrecte"""
        response = self.app.post(
            '/feedback',
            data=json.dumps({
                'tweet': 'This service is terrible!',
                'predicted_sentiment': 'Positif',
                'is_correct': False
            }),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        
        # Le feedback négatif devrait envoyer une trace à Application Insights
        mock_send_to_appinsights.assert_called_once_with(
            'This service is terrible!', 'Positif', is_incorrect=True
        )
        
    def test_feedback_missing_fields(self):
        """Test de l'endpoint de feedback avec des champs manquants"""
        response = self.app.post(
            '/feedback',
            data=json.dumps({
                'tweet': 'This service is terrible!'
                # predicted_sentiment manquant
            }),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
        
    def test_preprocess_tweet(self):
        """Test de la fonction de prétraitement des tweets"""
        test_cases = [
            (
                "I love @AirParadis! Their service is amazing! #greatservice https://example.com",
                "i love  their service is amazing greatservice "
            ),
            (
                "@user This flight was delayed by 2 hours :(",
                "this flight was delayed by  hours "
            ),
            (
                "#BadExperience with AirParadis - never flying with them again!",
                "badexperience with airparadis  never flying with them again"
            )
        ]
        
        for raw_tweet, expected_processed in test_cases:
            self.assertEqual(api_backup.preprocess_tweet(raw_tweet), expected_processed)

    @patch('requests.post')
    def test_send_to_appinsights_success(self, mock_post):
        """Test de l'envoi de télémétrie à Application Insights (succès)"""
        # Configuration du mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Configuration de la clé d'instrumentation
        orig_key = api_backup.APPINSIGHTS_INSTRUMENTATION_KEY
        api_backup.APPINSIGHTS_INSTRUMENTATION_KEY = "test-key"
        
        # Appel de la fonction
        api_backup.send_to_appinsights("test tweet", "Positif", is_incorrect=True)
        
        # Vérification que le mock a été appelé
        self.assertTrue(mock_post.called)
        
        # Restauration de la clé d'instrumentation
        api_backup.APPINSIGHTS_INSTRUMENTATION_KEY = orig_key
        
    @patch('requests.post')
    def test_send_to_appinsights_error(self, mock_post):
        """Test de l'envoi de télémétrie à Application Insights (erreur)"""
        # Configuration du mock
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        # Configuration de la clé d'instrumentation
        orig_key = api_backup.APPINSIGHTS_INSTRUMENTATION_KEY
        api_backup.APPINSIGHTS_INSTRUMENTATION_KEY = "test-key"
        
        # Appel de la fonction
        api_backup.send_to_appinsights("test tweet", "Positif", is_incorrect=True)
        
        # Vérification que le mock a été appelé
        self.assertTrue(mock_post.called)
        
        # Restauration de la clé d'instrumentation
        api_backup.APPINSIGHTS_INSTRUMENTATION_KEY = orig_key
        
    def test_send_to_appinsights_no_key(self):
        """Test de l'envoi de télémétrie sans clé d'instrumentation"""
        # Sauvegarde de la clé d'instrumentation
        orig_key = api_backup.APPINSIGHTS_INSTRUMENTATION_KEY
        api_backup.APPINSIGHTS_INSTRUMENTATION_KEY = ""
        
        # Appel de la fonction ne devrait pas lever d'exception
        try:
            api_backup.send_to_appinsights("test tweet", "Positif", is_incorrect=True)
            success = True
        except Exception:
            success = False
        
        # Restauration de la clé d'instrumentation
        api_backup.APPINSIGHTS_INSTRUMENTATION_KEY = orig_key
        
        self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()