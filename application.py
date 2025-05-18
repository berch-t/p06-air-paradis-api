#!/usr/bin/env python
# -*- coding: utf-8 -*-

from api import app
import os

# Point d'entrée pour Azure Web App
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)