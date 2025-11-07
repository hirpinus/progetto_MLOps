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

Il dataset scelto, ideale per l'addestramento sul social sentiment, è *kazanova/sentiment140*.

Viene scaricato dinamicamente da Kaggle usando la libreria kaggle hub nel sorgente /src/model_utils/dataset_loader.py

[https://www.kaggle.com/datasets/kazanova/sentiment140]

Utilizzando KAGGLE_USERNAME e KAGGLE_KEY come variabili d'ambiente contenute in secrets e importate in .github/workflows/CI_CD.yml .


Il modello viene addestrato utilizzando 

***src/model_trainer.py***

lanciabile anche manualmente con il comando:

*python3 src/model_trainer.py*


Nel suo flusso di esecuzione verrà utilizzato *model_utils/dataset_loader.py*

per il caricamento e il preprocessing del dataset, successivamente verrà riaddestrato il modello e verranno generate le metriche per prometheus e grafana.

Il modello addestrato, il vocabolario e il tokenizer verranno generati e salvati a runtime nella cartella *models*.


## Adattamento ed encoding del dataset

In ***model_utils/dataset_loader.py***

sono state prese in considerazione solo due colonne del dataset, contenenti i messaggi e le classi di scoring di appartenenza,

inizialmente impostato a 0=negative, 2=neutral, 4=positive, che per il nostro caso 

è stato ricodificato rispettivamente in 0,1,2. 


Il dataset originale viene importato a runtime da remoto nella cartella (creata sempre a runtime) 

my_datasets/Sentiment140_original

nome del file: training.1600000.processed.noemoticon.csv


Dopo il preprocessing il dataset viene salvato nel file sentiment140_ready.csv nella cartella 

my_datasets/Sentiment140_preprocessed

anch'essa generata a runtime nello spazio dell'applicazione.


I dati vengono splittati in set di training e test (proporzione 80%/20%).

Per carenza di risorse e per la dimenzione del dataset, si è optato per un sampling di righe casuali di dimensione 160.


Come vedremo nelle conclusioni, la quantità di dati selezionata non è ideale per portare a convergenza un modello ottimale, 

ma date le risorse limitate della macchina, è sufficiente per il nostro scopo dimostrativo MLOps.


# Lancio della predizione

 Il file ***src/main.py*** , lanciabile anche manualmente almeno dopo un primo addestramento e successiva generazione del modello,
 contiene tre frasi di esempio e utilizza la classe ***SentimentPredictor*** (file ***src/sentiment_predict.py***) .
 

SentimentPredictor è una classe in cui vengono caricati il modello ri-addestrato, il tokenizer e il vocabolario, ed espone due funzioni utili rispettivamente a controllare che siano stati generati correttamente i files del modello alla fine della fase di addestramento e a interrogare il modello per una predizione e a classificarne l'esito.


# Endpoints FastAPI e raccolta metriche

***src/app.py***
Viene caricato il modello, vengono inizializzati un Counter delle predizioni e un Gauge per la confidence delle predizioni, 

viene effettuata la mount di monitiraggio per l'app *sentiment_app* sulla cartella /metrics  generata a runtime e vengono definiti due endpoint:

- predict : per effettuare la predizione
- health : per controllare che il modello sia up and running


 # Test di integrazione

In ***tests/test_class.py*** sono esposte tre funzioni di test invocate da **pytest**:

*test_model_assets_exist()* verifica che siano stati generati e salvati correttamente i pesi del modello, il tokenizer, il vocabolario e il file di configurazione json

*test_can_load_and_predict()* verifica che il modello sia responsivo, che generi una predizione e che la confidence sia nel range previsto tra 0 e 1

*test_main_script_smoke_runs()* lancia, attraverso uno smoke test, *main.py* e verifichi che esegua senza errori, il che significa che il modello è up e running

# Pipeline CI/CD

Configurazione: ***.github/workflows/CI_CD.yml***

Vengono definiti i seguenti step:
- Esecuzione ad ogni push o pull request sul branch main
- Lettura variabili di ambiente: user e token per kaggle, user e token per huggingface, puntamento alla cartella src per il path di python (serve per gli import in fase di esecuzione automatica dei test su macchina remota)
- Installazione delle dipendenze da requirements.txt, esportato con pip freeze
- Configurazioni per le credenziali di Kaggle da utilizzare nel download del dataset nello step di addestramento lanciato da pipeline CI/CD
- Addestramento del modello eseguendo ***model_trainer.py***
- Upload dell'artifact del modello nella cartella models/ dell'ambiente virtuale della pipeline (serve per l'invocazione dei test negli step successivi)
- Push del modello su HuggingFace, solo se siamo sul branch main ed è presente un token valido. Repo: hirpinus/sentiment-analysis-profAI
- Lancio degli integration test presenti nella cartella tests

# Monitoraggio

Per il monitoraggio si è utilizzato grafana ( ***grafana/provisioning*** )
Per la raccolta delle metriche la scelta è ricaduta su prometheus, di utilizzo comune e consolidato con grafana ( ***prometheus/prometheus.yml***)

In *prometheus.yml* viene effettuato lo scraping sulla sentiment_app 

In ***grafana/provisioning/datasources/datasource.yml*** viene impostato prometheus come datasource per le rilevazioni


# Istruzioni di Configurazione

Clona il repository con il seguente comando bash:    
   
git clone https://github.com/hirpinus/progetto_MLOps 
cd progetto_MLOps    


Esegui l'installazione delle librerie necessarie con il seguente comando:

pip install -r requirements.txt


Considerato che chi installerà e configurerà per la prima volta il progetto lo farà su una macchina sulla quale non c'è già il modello,

è necessario lanciare a mano la prima volta l'addestramento, eseguendo il comando:

*python3 src/model_trainer.py*


Esegui da riga di comando:      
docker-compose up --build    

# Istruzioni di Esecuzione

Eseguire 

- L'applicazione sarà disponibile su http://localhost:5001.    
- Si può accedere a Grafana su http://localhost:3000 (credenziali predefinite: admin/admin).

Esempio di request per la predizione: 

curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d '{"text": "FastAPI è incredibilmente veloce e facile da usare!"}' 

# Conclusioni

Nei vari test, si nota (anche da grafana) che il modello genera sempre predizioni Neutre con un alto grado di confidenza,
indipendentemente dall'input, poichè è stato addestrato solo su 160 righe del dataset e questo ha comportato una bassa accuracy.

Il dataset, per quanto molto utile ed efficace, contiene 1,6 milioni di righe, quindi per un addestramento più efficace è bene usare una macchina più potente rispetto a quella di CodeSpaces.


