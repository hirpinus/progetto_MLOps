from model_utils import dataset_loader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os

BASE_PATH = Path(__file__).resolve().parents[1]  # root del progetto
KAGGLE_DATASET_NAME = "kazanova/sentiment140"
DATASET_ORIGINAL_PATH = BASE_PATH / "my_datasets" / "Sentiment140_original"
DATASET_ORIGINAL_FILENAME = "training.1600000.processed.noemoticon.csv"
DATASET_PROCESSED_PATH = BASE_PATH / "my_datasets" / "Sentiment140_preprocessed"
DATASET_PROCESSED_FILENAME = "sentiment140_ready.csv"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_SAVE_PATH = BASE_PATH / "models"

KAGGLE_USERNAME_ENV = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY_ENV = os.getenv("KAGGLE_KEY")


def train_model():

    dataset_loader.download_kaggle_dataset(KAGGLE_DATASET_NAME, 
                                           DATASET_ORIGINAL_PATH, 
                                           DATASET_ORIGINAL_FILENAME,
                                           KAGGLE_USERNAME_ENV, 
                                           KAGGLE_KEY_ENV)

    preprocessed_dataset_path = dataset_loader.dataset_preprocess(DATASET_ORIGINAL_PATH,
                                                         DATASET_ORIGINAL_FILENAME,
                                                         DATASET_PROCESSED_PATH,
                                                         DATASET_PROCESSED_FILENAME,
                                                         sample_size=160)
                                            #per demo e poche risorse usiamo solo un campione di esempio


    dataset = dataset_loader.load_and_split_dataset(DATASET_PROCESSED_PATH,
                                                    DATASET_PROCESSED_FILENAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Tokenizza il dataset
    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_dataset = dataset.map(preprocess, batched=True)

   #Parametri di training — compatibilità con versioni di transformers che non accettano evaluation_strategy
    try:
        training_args = TrainingArguments(
            output_dir="./logs/results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=2, #poche epoche per ottimizzare le risorse
            weight_decay=0.01,
        )
    except TypeError:
        training_args = TrainingArguments(
            output_dir="./logs/results",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=2,
            weight_decay=0.01,
        )
        # se l'attributo è supportato, impostalo dinamicamente
        if hasattr(training_args, "evaluation_strategy"):
            setattr(training_args, "evaluation_strategy", "epoch")


    # Trainer Huggingface
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    # Lancia il fine-tuning
    trainer.train()

    #Salva il nuovo modello riaddestrato
    trainer.save_model(MODEL_SAVE_PATH)

    #Dobbiamo salvare anche il modello pretrained per le configurazioni future
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

train_model()