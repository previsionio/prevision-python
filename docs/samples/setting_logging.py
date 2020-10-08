# [Prevision.io - Python SDK]
# Sample: Setting logging
#
# ---------------------------------------------------------------
# Author: Mina Pêcheux
# Date: July 2020
# ===============================================================

import previsionio as pio

# CHANGE LOGGING LEVEL ------------------------------------------
pio.verbose(True, debug=True) # (add event_log=True
                                # for events logging)

# CLIENT INITIALIZATION -----------------------------------------
url = """https://<your instance>.prevision.io"""
token = """<your token>"""
pio.client.init_client(url, token)

# TESTING LOGS --------------------------------------------------
# fetching a dataset from the platform
dataset = pio.Dataset.from_name('helloworld')

# fetching a usecase from the platform
uc = pio.Supervised.from_name('helloworld classif')
