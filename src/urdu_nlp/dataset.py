"""Dataset management, stratified 80/10/10 splitting, and near-duplicate leakage prevention.

References: TRD §3 & PRD §5, §6.
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_token_jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Computes Jaccard similarity between two token sets."""
    union = len(tokens_a | tokens_b)
    if union == 0:
        return 0.0
    return len(tokens_a & tokens_b) / union


# Representative benchmark corpus of Roman Urdu and Nastaliq Urdu reviews
# with explicit tags for error slice evaluation: negation, sarcasm, and code-switching
BENCHMARK_RAW_REVIEWS = [
    # Positive Roman Urdu
    ("ye product bohat zabardast hai quality bht achi hai", "positive", "standard"),
    ("bht behtareen service time pe delivery mil gayi shukriya", "positive", "standard"),
    ("bohat achaa phone hai battery timing kamaal ki hai", "positive", "standard"),
    ("zabardast packaging aur fast shipping highly recommended", "positive", "code_switch"),
    ("camera quality is awesome bht pasand aya mujhy", "positive", "code_switch"),
    ("cheez original hai aur paise wasool hain", "positive", "standard"),
    ("dil khush ho gaya dekh kar bilkul same as shown", "positive", "standard"),
    ("superb quality sound bohat loud and clear hai", "positive", "code_switch"),
    ("bht he umda material hai dobara zaroor order karunga", "positive", "standard"),
    ("100% recommended product bht theek kaam kar raha hai", "positive", "standard"),
    # Negative Roman Urdu
    ("bilkul fazool cheez hai paise zaya ho gaye", "negative", "standard"),
    ("bohat khrab experience raha product toota hua tha", "negative", "standard"),
    ("bakwas service hai kabhi order mat karna yahan se", "negative", "standard"),
    ("quality intehai ghatiya hai aur wapis b nai le rahe", "negative", "standard"),
    ("completely useless product koi faida nai lene ka", "negative", "code_switch"),
    ("fraud company hai broken item send kia hai", "negative", "code_switch"),
    ("worst customer support call he attend nai karte", "negative", "code_switch"),
    ("rang utar gaya pehle he din bohat bura experience", "negative", "standard"),
    ("size bilkul galat bheja hai irresponsible seller", "negative", "code_switch"),
    ("cheating seller hai fake product dispatch kar dia", "negative", "code_switch"),
    # Neutral Roman Urdu
    ("product normal hai na bohat acha na bohat bura", "neutral", "standard"),
    ("theek thaak hai price k hisab se guzara hai", "neutral", "standard"),
    ("delivery time pe thi product is average", "neutral", "code_switch"),
    ("okayish quality packing simple thi", "neutral", "code_switch"),
    ("kam chal jata hai par premium nahi hai", "neutral", "standard"),
    ("jo order kia wohi aya par expected se thora chota hai", "neutral", "standard"),
    ("guzara karta hai price reasonable thi", "neutral", "standard"),
    ("delivery late hui lekin product theek hai", "neutral", "standard"),
    # Negation Slice (Explicit negation polarity inversions)
    ("ye mobile bilkul acha nahi hai", "negative", "negation"),
    ("quality achi nai thi bht jaldi kharab ho gaya", "negative", "negation"),
    ("camera result theek nahi hai blur ata hai", "negative", "negation"),
    ("delivery time pe nahi pohnchi bohat der kardi", "negative", "negation"),
    ("mujhe yeh design pasand nahi aya", "negative", "negation"),
    ("is price mein koi faida nahi lene ka", "negative", "negation"),
    ("koi masla nahi hai sab theek se chal raha hai", "positive", "negation"),
    ("bilkul b bura nahi hai expected se acha nikla", "positive", "negation"),
    ("shikayat ka moqa nahi mila bht fast service", "positive", "negation"),
    ("koi complaint nahi hai highly satisfied", "positive", "negation"),
    # Sarcasm Slice (Positive words used with sarcastic negative intent)
    ("wah kya kamal ki service hai 2 hafte baad toota hua bheja", "negative", "sarcasm"),
    ("bohat zabardast packaging thi poora box daba hua tha", "negative", "sarcasm"),
    ("shukriya apka khrab cheez bhej kar paise lootne ka", "negative", "sarcasm"),
    ("itni behtareen quality hai k pehle ghante me he band ho gaya", "negative", "sarcasm"),
    ("bohat mehrbani itna late parcel pohanchane ki", "negative", "sarcasm"),
    # Nastaliq Urdu Script Reviews
    ("یہ موبائل بہت بہترین ہے بیٹری ٹائمنگ شاندار ہے۔", "positive", "standard"),
    ("انتہائی شاندار پروڈکٹ ہے اور ڈیلیوری بھی وقت پر ہوئی۔", "positive", "standard"),
    ("کوالٹی بہت عمدہ ہے اور قیمت کے حساب سے بہترین ہے۔", "positive", "standard"),
    ("بہت ہی خراب چیز ہے پیسے ضائع ہو گئے۔", "negative", "standard"),
    ("بالکل فضول پروڈکٹ ہے کام ہی نہیں کر رہا۔", "negative", "standard"),
    ("نہایت گھٹیا سروس ہے کوئی بھی یہ نہ خریدے۔", "negative", "standard"),
    ("یہ بالکل بھی اچھا نہیں ہے جلدی ٹوٹ گیا۔", "negative", "negation"),
    ("پروڈکٹ ٹھیک نہیں ہے شکایت کے باوجود جواب نہیں ملا۔", "negative", "negation"),
    ("عام سی چیز ہے قیمت مناسب ہے۔", "neutral", "standard"),
    ("درمیانی کوالٹی ہے زیادہ امیدیں نہ رکھیں۔", "neutral", "standard"),
]


def create_stratified_dataset(
    output_dir: Path | str = "data/processed",
    seed: int = 42,
    train_ratio: float = 0.80,
    dev_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> tuple[Path, Path, Path]:
    """Generates stratified 80/10/10 train/dev/test splits with near-duplicate leak prevention."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # 1. Expand dataset synthetically with realistic variations to simulate ~600 reviews
    corpus: list[dict] = []
    doc_id = 0

    prefixes = ["", "bhai ", "salam ", "yaar ", "really ", "actually "]
    suffixes = ["", " recommend karunga", " plz check karlein", " thanks", " شکریہ"]

    # Representative spelling variation maps to reflect real social media noise
    spelling_noise = {
        "bohat": ["bht", "bohut", "bohattt", "bhot", "bhoot", "bohaaat"],
        "nahi": ["nhi", "nai", "nhe", "nahii", "nh"],
        "acha": ["achha", "achaa", "achhaaa", "achah", "achaay"],
        "kharab": ["khrab", "khraab", "kharaab", "kharaaab"],
        "zabardast": ["zbrdst", "zabardst", "zbrdast", "zabardaaast"],
        "behtareen": ["behtreen", "bhtareen", "bhtreen", "behtareeem"],
        "bakwas": ["bkwas", "bakwaas", "bkwass", "bakwaaas"],
        "pasand": ["psand", "passand", "psnd"],
        "theek": ["thik", "thk", "thek", "theeek"],
        "shukriya": ["shukrya", "sukriya", "shukriyaa"],
        "cheez": ["chez", "chiz", "cheeez"],
        "paise": ["paisay", "pasy", "pesy", "paasay"],
        "dobara": ["dubara", "dobaara", "dobarah"],
        "wapis": ["wapsi", "waps", "waapis"],
        "faida": ["fayda", "faydha", "faaida"],
        "masla": ["msla", "maslah", "maslaa"],
    }

    for text, label, slice_tag in BENCHMARK_RAW_REVIEWS:
        for p in prefixes:
            for s in suffixes[:2]:
                raw_base = f"{p}{text}{s}".strip()
                # 1. Base clean sentence
                tokens_clean = set(raw_base.lower().split())
                corpus.append(
                    {
                        "doc_id": doc_id,
                        "text": raw_base,
                        "label": label,
                        "slice_tag": slice_tag,
                        "tokens": tokens_clean,
                    }
                )
                doc_id += 1

                # 2. Variant with informal spelling noise and elongations
                words = raw_base.split()
                noisy_words = []
                for w in words:
                    w_lower = w.lower()
                    if w_lower in spelling_noise and rng.random() < 0.70:
                        noisy_words.append(rng.choice(spelling_noise[w_lower]))
                    else:
                        noisy_words.append(w)
                noisy_text = " ".join(noisy_words)
                tokens_noisy = set(noisy_text.lower().split())
                corpus.append(
                    {
                        "doc_id": doc_id,
                        "text": noisy_text,
                        "label": label,
                        "slice_tag": slice_tag,
                        "tokens": tokens_noisy,
                    }
                )
                doc_id += 1

    # 2. Near-Duplicate Clustering (Jaccard >= 0.90 must be in same split to prevent leakage)
    clusters: list[list[dict]] = []
    assigned = set()

    for i, item_a in enumerate(corpus):
        if i in assigned:
            continue
        cluster = [item_a]
        assigned.add(i)
        for j, item_b in enumerate(corpus[i + 1 :], start=i + 1):
            if j not in assigned and item_a["label"] == item_b["label"]:
                jacc = compute_token_jaccard(item_a["tokens"], item_b["tokens"])
                if jacc >= 0.90:
                    cluster.append(item_b)
                    assigned.add(j)
        clusters.append(cluster)

    # 3. Stratified allocation of clusters by label
    train_items, dev_items, test_items = [], [], []

    for label in ("positive", "negative", "neutral"):
        label_clusters = [c for c in clusters if c[0]["label"] == label]
        random.Random(seed).shuffle(label_clusters)

        n_c = len(label_clusters)
        n_train = max(1, int(round(n_c * train_ratio)))
        n_dev = max(1, int(round(n_c * dev_ratio)))

        c_train = label_clusters[:n_train]
        c_dev = label_clusters[n_train : n_train + n_dev]
        c_test = label_clusters[n_train + n_dev :]

        for cl in c_train:
            train_items.extend(cl)
        for cl in c_dev:
            dev_items.extend(cl)
        for cl in c_test:
            test_items.extend(cl)

    # 4. Invariant assertion: zero duplicate text across splits
    train_texts = {r["text"] for r in train_items}
    dev_texts = {r["text"] for r in dev_items}
    test_texts = {r["text"] for r in test_items}

    assert len(train_texts & dev_texts) == 0, "Data leakage between train and dev!"
    assert len(train_texts & test_texts) == 0, "Data leakage between train and test!"
    assert len(dev_texts & test_texts) == 0, "Data leakage between dev and test!"

    # 5. Write split CSV files
    train_file = output_dir / "train.csv"
    dev_file = output_dir / "dev.csv"
    test_file = output_dir / "test.csv"

    fieldnames = ["doc_id", "text", "label", "slice_tag"]

    def write_csv(path: Path, rows: list[dict]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    write_csv(train_file, train_items)
    write_csv(dev_file, dev_items)
    write_csv(test_file, test_items)

    logger.info(
        "UrduNLP splits written: Train=%d, Dev=%d, Test=%d (Total=%d)",
        len(train_items),
        len(dev_items),
        len(test_items),
        len(train_items) + len(dev_items) + len(test_items),
    )
    return train_file, dev_file, test_file


if __name__ == "__main__":
    create_stratified_dataset()
