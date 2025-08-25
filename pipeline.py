import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import re
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
MODEL_NAME = "bert-base-uncased"
SONG_QUESTIONS = [
    "Does this song contain any violent themes, such as references to guns, killing, or physical aggression? Example: Does the song describe or promote physical violence, like fighting or shootings?",
    "Are there any explicit lyrics or bad words used in this song that might be considered offensive or inappropriate? Example: Does the song use language commonly recognized as profanity or derogatory terms?",
    "Is the overall content of this song suitable for children, considering its themes, language, and messages? Example: Are there elements in the song that could be deemed too mature or unsuitable for young listeners?",
    "Does this song explicitly mention weapons, such as guns, knives, or other similar items? Example: Are specific types of weapons described or glorified in the lyrics?",
    "Are the messages conveyed in this song positive and uplifting for children? Example: Does the song promote values like kindness, friendship, and positivity?",
    "Does this song include any sexual content, references to sexual behavior, or suggestive language? Example: Are there lyrics that explicitly or implicitly discuss sexual themes or experiences?",
    "Does this song offer any educational value, such as teaching the alphabet, basic math, or other learning content? Example: Are there educational segments in the song that could help children learn fundamental skills like the ABCs or counting?",
    "Does this song promote emotional resilience and social skills among children? Example: Does the song include themes of overcoming challenges or building friendships?"
]

YES_RESPONSES = [
    "Yes, this song contains violent themes, including references to guns, killing, or physical aggression, and is not suitable for children.",
    "Yes, this song includes explicit lyrics or bad words that might be considered offensive or inappropriate for young audiences.",
    "No, the overall content of this song is not suitable for children as it includes themes, language, and messages that are too mature or unsuitable for young listeners.",
    "Yes, this song explicitly mentions weapons, such as guns and knives, which could be disturbing or inappropriate for children’s entertainment.",
    "Yes, the messages conveyed in this song are positive and uplifting, promoting values like kindness, friendship, and positivity, beneficial for children.",
    "Yes, this song includes sexual content and references to sexual behavior or suggestive language, which are inappropriate for a child-friendly environment.",
    "Yes, this song offers significant educational value, including segments that teach the alphabet, basic math, and other learning content, making it both fun and educational for children.",
    "Yes, this song promotes emotional resilience and social skills, incorporating themes about overcoming challenges and building friendships, which are essential for children's development."
]

# 0: violent, 1: explicit/profanity, 2: overall_unsuitable, 3: weapons,
# 4: positive/uplifting, 5: sexual_content, 6: educational, 7: social/resilience
QUESTION_INTENTS = {
    0: "violence",
    1: "profanity",
    2: "overall_unsuitable",
    3: "weapons",
    4: "positive",
    5: "sexual",
    6: "educational",
    7: "social",
}
# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def tsne_plot(data, plot):
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # ensure this runs on cpu
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    elif isinstance(data, (list, tuple)) and len(data) > 0 and torch.is_tensor(data[0]):
        data = torch.stack(data).detach().cpu().numpy()

    # shrink the vectors to 3D using t-SNE
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(50, data.shape[0] - 1))
    data_3d = tsne.fit_transform(data)

    # plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # assign colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(data_3d)))
    for index, point in zip(range(len(data_3d)), data_3d):
        ax.scatter(point[0], point[1], point[2], color=colors[index], label=f"{plot} {index+1}")

    # labels and titles
    ax.set_xlabel("TSNE Component 1")
    ax.set_ylabel("TSNE Component 2")
    ax.set_zlabel("TSNE Component 3")
    plt.title("3D t-SNE Visualization of "+plot+" Embeddings")
    plt.legend(title=plot + " Index", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()

def text_to_embeddings(list_of_text, max_input=512):
    data_token_index = tokenizer.batch_encode_plus(list_of_text, add_special_tokens=True, padding=True, truncation=True, max_length=max_input)
    question_embeddings = aggregate_embeddings(data_token_index["input_ids"], data_token_index["attention_mask"])
    return question_embeddings

def process_song(song):
    # remove line breaks
    new_song = re.sub(r'\n', ' ', song)
    new_song = [new_song.replace("\'", "")]
    return new_song

def RAG_QA_dot_product(embeddings_questions, embeddings, n_responses=3):
    """Retrieve the top rankings of the dot product between the questions embeddings and the provided embeddings"""
    dot_product = embeddings_questions @ embeddings.T

    # reshape the dot product results to a 1D tensor and sort
    dot_product = dot_product.reshape(-1)
    sorted_indices = torch.argsort(dot_product, descending=True)
    sorted_indices = sorted_indices.tolist()

    # print the top results from the sorted list
    for index in sorted_indices[:n_responses]:
        print(YES_RESPONSES[index])

def rank_questions_dot(embeddings_questions: torch.Tensor,
                       song_embedding: torch.Tensor) -> list[int]:
    """
    Return a ranked list of question indices by dot product similarity.
    embeddings_questions: [Q, H]
    song_embedding:       [1, H]
    """
    scores = (embeddings_questions @ song_embedding.T).reshape(-1)  # [Q]
    ranked = torch.argsort(scores, descending=True).tolist()
    return ranked

def precision_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if k <= 0: return 0.0
    topk = ranked[:k]
    hits = sum(1 for i in topk if i in relevant)
    return hits / k

def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant: return 0.0
    topk = ranked[:k]
    hits = sum(1 for i in topk if i in relevant)
    return hits / len(relevant)

def reciprocal_rank(ranked: list[int], relevant: set[int]) -> float:
    for idx, qid in enumerate(ranked, start=1):  # ranks are 1-based
        if qid in relevant:
            return 1.0 / idx
    return 0.0
# ---------------------------------------------------------------------------
# 1. Load model and tokenizer
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
bert_model.eval()

def aggregate_embeddings(input_ids, attention_masks, bert_model=bert_model):
    """take a bunch of variable-length token sequences (lyrics, questions, responses, etc.)
    and turn each one into a single fixed-length vector so you can compare them consistently."""
    mean_embeddings = []
    print("number of inputs", len(input_ids))
    for input_id, mask in tqdm(zip(input_ids, attention_masks)):
        input_id_tensor = torch.tensor([input_id]).to(DEVICE) # shape [1, T] per sequence
        mask_tensor = torch.tensor([mask]).to(DEVICE) # 1 for real tokens, 0 for paddings

        with torch.no_grad():
            # obtain word embeddings
            word_embeddings = bert_model(input_id_tensor, attention_mask=mask_tensor)[0].squeeze(0) # convert to word embeddings

            # filter out the embeddings at positions where the mask is zero
            valid_embeddings_mask = mask_tensor[0] != 0 # turn (1, 0) to (True, False)
            valid_embeddings = word_embeddings[valid_embeddings_mask, :] # drop paddings

            # compute the mean of the filtered embeddings
            mean_embedding = valid_embeddings.mean(dim=0) # shape: [T]
            mean_embeddings.append(mean_embedding.unsqueeze(0)) # unsqueeze for concat afterwards

    # concat the mean embeddings from all sequences in the batch
    aggregated_mean_embeddings = torch.cat(mean_embeddings)
    return aggregated_mean_embeddings

# ---------------------------------------------------------------------------
# 2. ETL
# ---------------------------------------------------------------------------
# embed questions and yes responses to them
embeddings_questions = text_to_embeddings(SONG_QUESTIONS)
embeddings_responses = text_to_embeddings(YES_RESPONSES)

# visualisation
tsne_plot(embeddings_questions, "Question")
tsne_plot(embeddings_responses, "Response")

songs = {}
# songs — fetch some songs online
songs["Bullet in the Head"] = """
This time the bullet cold rocked ya
A yellow ribbon instead of a swastika
Nothin’ proper about ya propaganda
Fools follow rules when the set commands ya
Said it was blue
When ya blood was read
That’s how ya got a bullet blasted through ya head
Blasted through ya head
Blasted through ya head
I give a shout out to the living dead
Who stood and watched as the feds cold centralized
So serene on the screen
You were mesmerised
Cellular phones soundin’ a death tone
Corporations cold
Turn ya to stone before ya realise
They load the clip in omnicolour
Said they pack the 9, they fire it at prime time
Sleeping gas, every home was like Alcatraz
And mutha fuckas lost their minds
Just victims of the in-house drive-by
They say jump, you say how high
Just victims of the in-house drive-by
They say jump, you say how high
Run it!
"""

songs["Sesame Street"] = """
Sunny day
Sweepin' the clouds away
On my way to where the air is sweet
Can you tell me how to get
How to get to Sesame Street?

Come and play
Everything's A-okay
Friendly neighbors there
That's where we meet
Can you tell me how to get
How to get to Sesame Street?

It's a magic carpet ride
Every door will open wide
To happy people like you
Happy people like
What a beautiful

Sunny day
Sweepin' the clouds away
On my way to where the air is sweet
Can you tell me how to get
How to get to Sesame Street?
How to get to Sesame Street?
How to get to Sesame Street?
How to get to Sesame Street?
How to get to Sesame Street?
"""

songs["Barney"] = """
Barney is a dinosaur from our imagination
And when he's tall
He's what we call a dinosaur sensation
Barney's friends are big and small
They come from lots of places
After school they meet to play
And sing with happy faces
Barney shows us lots of things
Like how to play pretend
ABC's, and 123's
And how to be a friend
Barney comes to play with us
Whenever we may need him
Barney can be your friend too
If you just make-believe him!
"""

songs["Straight Outta Compton Lyrics"] = """
Straight outta Compton, another crazy-ass nigga
More punks I smoke, yo, my rep gets bigger
I'm a bad motherfucker, and you know this
But the pussy-ass niggas won't show this
But I don't give a fuck, I'ma make my snaps
If not from the records, from jacking or craps
Just like burglary, the definition is jacking
And when I'm legally armed it's called packing
Shoot a motherfucker in a minute
I find a good piece of pussy and go up in it
So if you're at a show in the front row
I'ma call you a bitch or dirty-ass ho
You'll probably get mad like a bitch is supposed to
But that shows me, slut, you're not opposed to
A crazy motherfucker from the street
Attitude legit, ‘cause I'm tearing up shit
MC Ren controls the automatic
For any dumb motherfucker that starts static
"""

embeddings = {}
for name, song in songs.items():
    song = process_song(song)
    embedding = text_to_embeddings(song)
    embeddings[name] = embedding

all_embeddings = torch.cat([embedding for embedding in embeddings.values()], dim=0)
tsne_plot(all_embeddings, "Song")

# ---------------------------------------------------------------------------
# 3. Retrieval
# ---------------------------------------------------------------------------
for name, embedding in embeddings.items():
    print('\n' + '-' * 10 + name + '-' * 10)
    RAG_QA_dot_product(embeddings_questions, embedding)

# ---------------------------------------------------------------------------
# 4. Ground truth & evaluation
# ---------------------------------------------------------------------------
# Ground-truth "relevant questions" per song (indices into SONG_QUESTIONS)
GROUND_TRUTH = {
    "Bullet in the Head":            {0, 1, 2, 3},          # violence, profanity, not suitable, weapons
    "Straight Outta Compton Lyrics": {0, 1, 2, 3, 5},       # violence, profanity, not suitable, weapons, sexual
    "Sesame Street":                 {4, 6},                # positive, educational
    "Barney":                        {4, 6},                # positive, educational
}

def evaluate_retrieval(embeddings_questions: torch.Tensor,
                       song_embeddings: dict[str, torch.Tensor],
                       ground_truth: dict[str, set[int]],
                       ks=(1, 5)) -> None:
    """
    Prints per-song P@k / R@k / RR and macro averages across the dataset.
    `song_embeddings` is your dict {name: [1, H] tensor}
    """
    # store per-song metrics
    results = {}
    macro = {f"P@{k}": 0.0 for k in ks} | {f"R@{k}": 0.0 for k in ks}
    macro["MRR"] = 0.0

    n = 0
    for name, emb in song_embeddings.items():
        # skip songs without labels
        if name not in ground_truth:
            continue
        relevant = ground_truth[name]
        ranked = rank_questions_dot(embeddings_questions, emb)

        # per-song metrics
        metrics = {}
        for k in ks:
            metrics[f"P@{k}"] = precision_at_k(ranked, relevant, k)
            metrics[f"R@{k}"] = recall_at_k(ranked, relevant, k)
        metrics["MRR"] = reciprocal_rank(ranked, relevant)

        results[name] = metrics

        # accumulate for macro
        for k in ks:
            macro[f"P@{k}"] += metrics[f"P@{k}"]
            macro[f"R@{k}"] += metrics[f"R@{k}"]
        macro["MRR"] += metrics["MRR"]
        n += 1

    # averages
    if n > 0:
        for k in ks:
            macro[f"P@{k}"] /= n
            macro[f"R@{k}"] /= n
        macro["MRR"] /= n

    print("\n" + "-" * 10 + "Retrieval Evaluation (dot product)" + "-" * 10)
    for name, m in results.items():
        line = [f"{name}"]
        for k in ks:
            line.append(f"P@{k}={m[f'P@{k}']:.3f}")
            line.append(f"R@{k}={m[f'R@{k}']:.3f}")
        line.append(f"MRR={m['MRR']:.3f}")
        print("  " + " | ".join(line))

    print("\n"+ "-" * 10 + "Macro Averages" + "-" * 10)
    macro_line = []
    for k in ks:
        macro_line.append(f"P@{k}={macro[f'P@{k}']:.3f}")
        macro_line.append(f"R@{k}={macro[f'R@{k}']:.3f}")
    macro_line.append(f"MRR={macro['MRR']:.3f}")
    print("  " + " | ".join(macro_line))

# run eval
evaluate_retrieval(embeddings_questions, embeddings, GROUND_TRUTH, ks=(1, 5))
