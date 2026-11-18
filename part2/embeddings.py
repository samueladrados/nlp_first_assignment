
import numpy as np
from pathlib import Path
from collections import Counter

from part2.clslib import PretrainedEmbeddings, BasicTokenizer
from part2.experiment import train


def load_embeddings(option="wikigiga"):

    models = {
        "wikigiga": {
            "filename": "glove.2024.wikigiga.50d.zip",
            "dims": 50,
            "skip_first": False
        },
        "twitter": {
            "filename": "glove.twitter.27B.50d.txt.gz",
            "dims": 50,
            "skip_first": True
        }
    }

    if option not in models:
        raise ValueError(f"Unknown embedding model: {option}")

    config = models[option]

    path = Path(__file__).resolve().parent / "models" / config["filename"]

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return PretrainedEmbeddings(
        str(path),
        config["dims"],
        skip_first_line=config["skip_first"]
    )

# creating tokenizer compatible with the embeddings
def create_tokenizer(embedding_model, lowercase=True, remove_punct=False, remove_stopwords=False):
    """
    tokenizing texts in a way coherent with the definition of a token in the embedding model
    """
    return BasicTokenizer(
        lowercase=lowercase,
        remove_punct=remove_punct,
        remove_stopwords=remove_stopwords,
        language="english",
        vocab=embedding_model.vocab
    )

# calculate vocabulary coverage analysis code
# How many words in your tweets are actually known to the pretrained embedding model?
def vocab_coverage(texts, tokenizer, emb_model):
    counts = Counter()
    for text in texts:
        counts.update(tokenizer.tokens(text))

    total_tokens = sum(counts.values())
    unique_tokens = len(counts)

    known_tokens = sum(
        count for word, count in counts.items()
        if word in emb_model.vocab
    )

    known_unique = sum(
        1 for word in counts
        if word in emb_model.vocab
    )

    unknown_words = [
        (word, count)
        for word, count in counts.items()
        if word not in emb_model.vocab
    ]

    unknown_words.sort(key=lambda x: x[1], reverse=True)

    return {
        "token_coverage": known_tokens / total_tokens if total_tokens else 0,
        "unique_coverage": known_unique / unique_tokens if unique_tokens else 0,
        "top_unknown": unknown_words[:20]
    }

def cos_similarity(u, v):  
    """
    Return the cosine similarity (a real number) of the two words in the embedding space of our model.
    """      
    u = np.asarray(u)
    v = np.asarray(v)

    dot_p = u @ v
    magnitude = np.linalg.norm(u) * np.linalg.norm(v)

    if magnitude == 0:
        raise ValueError("magnitude is zero, cosine similarity is undefined")
    
    cosine_sim = dot_p / magnitude
    return float(cosine_sim)

def topk_ids(v, E, k=10):    
    """
    word embedding: a D-dimensional vector
    E: our model's embedding matrix as a numpy array of shape [V, D]
    k: how many nearest neighbours we want to find

    Return a list with the k word ids that are nearest to the input word embedding v
     in embedding space according to cosine similarity. 
    """   
    v = np.asarray(v)
    E = np.asarray(E)

    dot_products = E @ v

    #magnitudes
    v_norm = np.linalg.norm(v)
    E_norms = np.linalg.norm(E, axis=1)

    denominators = E_norms * v_norm

    #to avoid zero divisions
    similarities = np.divide(
        dot_products,
        denominators,
        out=np.full(len(E), -np.inf, dtype=float),
        where=denominators != 0
    )

    top_ids = np.argsort(similarities)[::-1][:k]

    return top_ids.tolist()

def train_embed_pool(
    train_data,
    val_data,
    tokenizer,
    embedding_model,
    save_to="results/embed_pool/baseline",
    enc_kwargs=None,
    regressor_kwargs=None,
    opt_kwargs=None,
    device="cpu"
):
    """
    training an embed-and-pool classifier for
    tweetEval emotion classification.

    reuses the provided part2.experiment.train().
    """

    if enc_kwargs is None:
        enc_kwargs = {
            "freeze_embeddings": True,
            "average_pool": True
        }

    if regressor_kwargs is None:
        regressor_kwargs = {
            "hidden_size1": 50,
            "hidden_size2": 50,
            "dropout": 0.1
        }

    if opt_kwargs is None:
        opt_kwargs = {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "num_epochs": 20
        }

    #the tutorial's train() modifies some of the dictionaries
    enc_kwargs = enc_kwargs.copy()
    regressor_kwargs = regressor_kwargs.copy()
    opt_kwargs = opt_kwargs.copy()

    # results directory 
    Path(save_to).parent.mkdir(
        parents=True,
        exist_ok=True)

    # train using the provided implementation
    metrics, best_epoch = train(
        training_set=train_data,
        validation_set=val_data,
        tokenizer=tokenizer,
        emb_model=embedding_model,
        num_classes=4,               #since tweet eval has 4 classes
        enc_type="embed-and-pool",
        save_to=save_to,
        enc_kwargs=enc_kwargs,
        regressor_kwargs=regressor_kwargs,
        opt_kwargs=opt_kwargs,
        device=device
    )

    return metrics, best_epoch