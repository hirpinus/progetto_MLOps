# Monitoraggio della reputazione online di un'Azienda - Corso di MLOps e Machine Learning in Produzione
Master AI Engineering di ProfessionAI

Questo progetto implementa un'applicazione di analisi del sentiment utilizzando un modello ri-addestrato. Sfrutta Docker per la containerizzazione e include funzionalità di monitoraggio con Grafana e Prometheus per valutare continuamente le performance del modello e il sentiment rilevato

Tecnologie Utilizzate:

![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Grafana](https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white)
![JSON](https://img.shields.io/badge/JSON-000000?style=for-the-badge&logo=json&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![Python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)


# Scelta del modello e del dataset e addestramento
Il modello pre-trained usato è RoBERTa
[https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest]

Il dataset scelto, ideale per l'addestramento sul social sentiment, è *kazanova/sentiment140*

Scaricato dinamicamente da Kaggle usando la libreria kaggle hub nel sorgente /src/model_utils/dataset_loader.py

[https://www.kaggle.com/datasets/kazanova/sentiment140]

Utilizzando KAGGLE_USERNAME e KAGGLE_KEY come variabili d'ambiente contenute in secrets e importate in .github/workflows/CI_CD.yml 

Il modello viene addestrato utilizzando 

***src/model_trainer.py***

Lanciabile anche manualmente con il comando:

*python3 src/model_trainer.py*

Nel cui flusso di esecuzione verrà utilizzato *model_utils/dataset_loader.py*

per il caricamento e il preprocessing del dataset, successivamente verrà riaddestrato il modello e verranno generate le metriche per prometheus e grafana.

Il modello addestrato, il vocabolario e il tokenizer verranno generati e salvati a runtime nella cartella *models*.


## Adattamento ed encoding del dataset

In ***model_utils/dataset_loader.py***

Sono state prese in considerazione solo due colonne del dataset, contenenti i messaggi e lo scoring,

inizialmente impostato a 0=negative, 2=neutral, 4=positive, che per il nostro caso 

è stato ricodificato rispettivamente in 0,1,2. 

Il dataset originale viene importato a runtime da remoto nella cartella (creata sempre a runtime) 

my_datasets/Sentiment140_original

nome del file: training.1600000.processed.noemoticon.csv

Dopo il preprocessing il dataset viene salvato nel file sentiment140_ready.csv nella cartella 

my_datasets/Sentiment140_preprocessed

anch'essa generata a runtime nello spazio dell'applicazione.


I dati vengono splittati in set di training e test (proporzione 80%/20%)

Per carenza di risorse e per la dimenzione del dataset, si è optato per un sampling di righe casuali di dimensione 160.

Come vedremo nelle conclusioni, la quantità di dati selezionata non è ideale per portare a convergenza un modello ottimale, 

ma date le risorse limitate della macchina, è sufficiente per il nostro scopo dimostrativo MLOps.

# Lancio della predizione

 Il file ***src/main.py*** , lanciabile anche manualmente almeno dopo un primo addestramento e generazione del modello,
 contiene tre frasi di esempio e utilizza la classe ***SentimentPredictor*** (file ***src/sentiment_predict.py***)

 SentimentPredictor è una classe in cui vengono caricati il modello ri-addestrato, il tokenizer e il vocabolario, ed espone due funzioni

 utili rispettivamente a controllare che siano stati generati correttamente i files del modello alla fine della fase di addestramento

 e a interrogare il modello per una predizione e a classificarne l'esito.

 # Test di integrazione

In ***tests/test_class.py*** sono esposte tre funzioni di test invocate da **pytest**:

*test_model_assets_exist()* verifica che siano stati generati e salvati correttamente i pesi del modello, il tokenizer, il vocabolario e il file di configurazione json

*test_can_load_and_predict()* verifica che il modello sia responsivo, che generi una predizione e che la confidence sia nel range previsto tra 0 e 1

*test_main_script_smoke_runs()* lancia, attraverso uno smoke test, *main.py* e verifichi che esegua senza errori, il che significa che il modello è up e running

# Pipeline CI/CD

# Istruzioni di Configurazione

Clona il repository con il seguente comando bash:    
   
git clone <repository-url>    
cd progetto_MLOps    

Crea ed esegui l'applicazione:    
(nella shell bash)   
docker-compose up --build    

---------------------------------Accedi all'applicazione:    
- L'applicazione sarà disponibile su http://localhost:5000.    
- Si può accedere a Grafana su http://localhost:3000 (credenziali predefinite: admin/admin).

---------------------------------Linee Guida per l'Utilizzo
Utilizza l'endpoint /predict per inviare testo per l'analisi del sentiment.

Monitora le metriche di performance e i risultati dell'analisi del sentiment tramite la dashboard di Grafana.

Architettura del Sistema

Docker: Containerizza l'applicazione, Grafana e Prometheus per una facile distribuzione e gestione.
Grafana: Visualizza le metriche di performance e i risultati dell'analisi del sentiment.
Prometheus: Monitora l'applicazione e raccoglie le metriche da visualizzare su Grafana.
