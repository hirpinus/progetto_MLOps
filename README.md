# Monitoraggio della reputazione online di un'Azienda - Corso di MLOps e Machine Learning in Produzione
Master AI Engineering di ProfessionAI

# progetto_MLOps

---------------------------------Panoramica
Questo progetto implementa un'applicazione di analisi del sentiment utilizzando un modello ri-addestrato. Sfrutta Docker per la containerizzazione e include funzionalità di monitoraggio con Grafana e Prometheus per valutare continuamente le performance del modello e il sentiment rilevato

---------------------------------Struttura del Progetto
progetto_MLOps
├── docker-compose.yml          #Definisce servizi, network e volumi per l'applicazione Docker
├── Dockerfile                  #Istruzioni per la creazione dell'immagine Docker per l'applicazione
├── grafana                     #Contiene i file di provisioning di Grafana
│   └── provisioning
│       ├── dashboards
│       │   ├── dashboard.yml   #Configura le impostazioni della dashboard di Grafana
│       │   └── sentiment_analysis_dashboard.json #Configurazione JSON per la dashboard di analisi del sentiment
│       └── datasources
│           └── datasource.yml   #Definisce la configurazione della fonte dati per Grafana
├── models                      #Contiene i file del modello di analisi del sentiment addestrato
│   └── sentiment_model
├── prometheus                  #File di configurazione di Prometheus
│   └── prometheus.yml          #Configurazione per il monitoraggio di Prometheus
├── requirements.txt            #Elenca le dipendenze Python necessarie per l'applicazione
├── src                         #Codice sorgente per l'applicazione
│   ├── app.py                  #Logica principale dell'applicazione e configurazione del web server
│   ├── sentiment_predict.py     #Definisce la classe SentimentPredictor
│   └── main.py                 #Punto di ingresso dell'applicazione
└── README.md                   #Documentazione per il progetto

---------------------------------Istruzioni di Configurazione

Clona il repository:    
(nella shell bash)     
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
