import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pickle, torch, torch.nn as nn
import numpy as np, pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict
import time, threading, io

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════════════
# ARTIFACTS (immuables)
# ══════════════════════════════════════════════════════════════════
with open("models/mytek_artifacts.pkl", "rb") as f:
    art = pickle.load(f)

sku2idx     = art["sku2idx"]
idx2sku     = art["idx2sku"]
CAT_GROUPS  = art["CAT_GROUPS"]
NUM_ITEMS   = art["NUM_ITEMS_MYTEK"]
MAX_SEQ_LEN = art["MAX_SEQ_LEN"]


# ══════════════════════════════════════════════════════════════════
# MODÈLE (immuable)
# ══════════════════════════════════════════════════════════════════
class SASRecMytek(nn.Module):
    def __init__(self, num_items, embed_dim=128, num_heads=4,
                 num_blocks=2, max_seq_len=50, dropout=0.2):
        super().__init__()
        self.item_emb    = nn.Embedding(num_items+1, embed_dim, padding_idx=0)
        self.pos_emb     = nn.Embedding(max_seq_len, embed_dim)
        self.emb_drop    = nn.Dropout(dropout)
        self.norm0       = nn.LayerNorm(embed_dim)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim*4, dropout=dropout,
            activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.norm_out    = nn.LayerNorm(embed_dim)
        self.register_buffer('causal_mask',
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool())

    def encode(self, input_seq):
        B, L = input_seq.shape
        pos  = torch.arange(L, device=input_seq.device).unsqueeze(0)
        x    = self.item_emb(input_seq) + self.pos_emb(pos)
        x    = self.emb_drop(self.norm0(x))
        x    = self.transformer(x, mask=self.causal_mask[:L,:L])
        return self.norm_out(x)[:, -1, :]

    def forward(self, input_seq):
        user_emb = self.encode(input_seq)
        logits   = user_emb @ self.item_emb.weight[1:].T
        return logits, user_emb


device = torch.device("cpu")
model  = SASRecMytek(NUM_ITEMS).to(device)
model.load_state_dict(torch.load("models/sasrec_mytek_best.pt", map_location=device))
model.eval()


# ══════════════════════════════════════════════════════════════════
# CONSTANTES MÉTIER
# ══════════════════════════════════════════════════════════════════
CAT_LEVELS         = ['cat_group', 'cat1', 'cat2', 'cat3', 'cat4']
CAT_LEVEL_FACTORS  = [0.15, 0.30, 0.55, 0.75, 1.0]
ACTION_BASE_WEIGHT = {'product_view': 1.0, 'add_to_cart': 3.0, 'category_view': 0.5}

NOISE_STD   = 0.08
TEMPERATURE = 1.5
EPSILON     = 0.15


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def _safe_float(val):
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    if s in ("", "nan", "none", "n/a", "N/A"):
        return None
    try:
        f = float(s)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _normalize_stock(raw):
    if not isinstance(raw, str) or not raw.strip():
        return "INCONNU"
    v = raw.strip().upper()
    if "STOCK"    in v: return "EN STOCK"
    if "ARRIVAGE" in v or "ARRIVÉE" in v: return "EN ARRIVAGE"
    if "EPUI"     in v or "ÉPUI"    in v: return "EPUISE"
    return v


def _is_valid_cat(val):
    if val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    return str(val).strip().upper() not in ("N/A", "", "NONE")


def _get_cat_hierarchy(meta):
    result = {}
    for i, field in enumerate(CAT_LEVELS):
        val = meta.get(field)
        if _is_valid_cat(val):
            result[i] = str(val).strip()
    return result


def _get_finest_cat(meta):
    h = _get_cat_hierarchy(meta)
    if not h:
        return None
    finest = max(h.keys())
    return finest, h[finest]


def _get_specific_cat(meta):
    res = _get_finest_cat(meta)
    return res[1] if res else "unknown"


def _is_promo(meta):
    ps = meta.get("prix_special")
    pn = meta.get("prix_normal")
    if ps is None or ps <= 0:
        return False
    if pn is not None and pn > 0:
        return ps < pn
    return True


def _passes_filters(meta, stock_filter, promo_only):
    if stock_filter and meta.get("stock") != "EN STOCK":
        return False
    if promo_only and not _is_promo(meta):
        return False
    return True


def _build_result(meta, rank):
    prix_special = meta["prix_special"]
    prix_normal  = meta["prix_normal"]
    if prix_normal and prix_special and prix_special >= prix_normal:
        prix_special = None
    return {
        "rank"        : rank,
        "sku"         : meta["sku"],
        "designation" : meta["designation"],
        "cat_group"   : meta["cat_group"],
        "categorie"   : meta["categorie"],
        "specific_cat": _get_specific_cat(meta),
        "prix"        : meta["prix"],
        "prix_normal" : prix_normal,
        "prix_special": prix_special,
        "en_promo"    : prix_special is not None,
        "stock"       : meta["stock"],
        "url"         : (f"https://www.mytek.tn/{meta['slug']}.html"
                         if meta.get("slug") else None),
    }


# ══════════════════════════════════════════════════════════════════
# BUILD CATALOG
# ══════════════════════════════════════════════════════════════════
def _build_catalog(df: pd.DataFrame) -> dict:
    col_map = {
        "mytek_référence"    : "sku",
        "mytek_désignation"  : "designation",
        "mytek_disponibilité": "stock",
        "mytek_prix"         : "prix",
        "mytek_prix_normal"  : "prix_normal",
        "mytek_prix_special" : "prix_special",
        "category1_name"     : "categorie",
        "category2_name"     : "cat1",
        "category3_name"     : "cat2",
        "category4_name"     : "cat3",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "mytek_url_article" in df.columns:
        df["slug"] = (df["mytek_url_article"]
                      .str.extract(r"/([^/]+)\.html$", expand=False)
                      .fillna(""))

    cat_name_to_group   = {k.lower(): v for k, v in CAT_GROUPS.items()}
    sku_meta            = {}
    cat_group_to_skuidx = defaultdict(list)
    level_cat_to_skuidx = defaultdict(list)

    for _, row in df.iterrows():
        sku = str(row.get("sku", "")).strip()
        if not sku or sku not in sku2idx:
            continue

        idx          = sku2idx[sku]
        prix         = _safe_float(row.get("prix"))
        prix_normal  = _safe_float(row.get("prix_normal")) or prix
        prix_special = _safe_float(row.get("prix_special"))

        if prix_special is not None and prix_normal is not None:
            if prix_special >= prix_normal:
                prix_special = None

        categorie = str(row.get("categorie", "")).strip()
        cat_group = cat_name_to_group.get(categorie.lower(), categorie)

        meta = {
            "sku"         : sku,
            "designation" : str(row.get("designation", "")).strip(),
            "categorie"   : categorie,
            "cat_group"   : cat_group,
            "cat1"        : str(row.get("cat1", "")).strip() or None,
            "cat2"        : str(row.get("cat2", "")).strip() or None,
            "cat3"        : str(row.get("cat3", "")).strip() or None,
            "prix"        : prix,
            "prix_normal" : prix_normal,
            "prix_special": prix_special,
            "stock"       : _normalize_stock(row.get("stock", "")),
            "slug"        : str(row.get("slug", "")).strip(),
        }

        sku_meta[idx] = meta
        cat_group_to_skuidx[cat_group].append(idx)
        for level_i, cat_val in _get_cat_hierarchy(meta).items():
            level_cat_to_skuidx[(level_i, cat_val)].append(idx)

    return {
        "sku_meta"            : sku_meta,
        "cat_group_to_skuidx" : cat_group_to_skuidx,
        "level_cat_to_skuidx" : level_cat_to_skuidx,
        "cat_name_to_group"   : cat_name_to_group,
        "product_count"       : len(sku_meta),
        "promo_count"         : sum(1 for m in sku_meta.values() if m["prix_special"] is not None),
        "loaded_at"           : time.time(),
    }


# ══════════════════════════════════════════════════════════════════
# BLUE / GREEN CATALOG SLOT
# ══════════════════════════════════════════════════════════════════
_active_catalog: dict | None = None
_catalog_lock                = threading.Lock()
_update_lock                 = threading.Lock()

_update_state = {"running": False, "started_at": None, "error": None}


def _get_active_catalog() -> dict:
    with _catalog_lock:
        if _active_catalog is None:
            raise RuntimeError("Aucun catalog actif. Uploadez un CSV via POST /update.")
        return _active_catalog


def _swap_catalog(new_catalog: dict):
    global _active_catalog
    with _catalog_lock:
        _active_catalog = new_catalog


def _do_update(csv_bytes: bytes):
    global _update_state
    _update_state = {"running": True, "started_at": time.time(), "error": None}
    try:
        df          = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, encoding="utf-8-sig")
        new_catalog = _build_catalog(df)
        _swap_catalog(new_catalog)
        _update_state["running"] = False
    except Exception as e:
        _update_state["running"] = False
        _update_state["error"]   = str(e)
        raise


# ══════════════════════════════════════════════════════════════════
# LOGIQUE RECOMMANDATION
# ══════════════════════════════════════════════════════════════════
def recommend(actions, top_k=10, category_boost=5.0,
              stock_filter=False, promo_only=False):

    catalog             = _get_active_catalog()
    sku_meta            = catalog["sku_meta"]
    cat_group_to_skuidx = catalog["cat_group_to_skuidx"]
    level_cat_to_skuidx = catalog["level_cat_to_skuidx"]
    cat_name_to_group   = catalog["cat_name_to_group"]

    RECENCY_MAX      = 3.0
    seq_idx          = []
    cart_skus        = set()
    viewed_skus      = set()
    cat_level_scores = defaultdict(float)
    n                = len(actions)

    for pos, (action_type, value) in enumerate(actions):
        recency_w = (1.0 + (RECENCY_MAX - 1.0) * pos / max(n - 1, 1) if n > 1 else 1.0)

        if action_type == "product_view":
            sku = str(value).strip()
            if sku in sku2idx:
                idx = sku2idx[sku]
                seq_idx.append(idx)
                viewed_skus.add(idx)
                total_w = recency_w * ACTION_BASE_WEIGHT["product_view"]
                for level_i, cat_val in _get_cat_hierarchy(sku_meta.get(idx, {})).items():
                    cat_level_scores[(level_i, cat_val)] += total_w

        elif action_type == "add_to_cart":
            sku = str(value).strip()
            if sku in sku2idx:
                idx = sku2idx[sku]
                seq_idx.extend([idx, idx, idx])
                cart_skus.add(idx)
                total_w = recency_w * ACTION_BASE_WEIGHT["add_to_cart"]
                for level_i, cat_val in _get_cat_hierarchy(sku_meta.get(idx, {})).items():
                    cat_level_scores[(level_i, cat_val)] += total_w

        elif action_type == "category_view":
            grp = cat_name_to_group.get(str(value).lower())
            if grp:
                pool = cat_group_to_skuidx.get(grp, [])
                if pool:
                    seq_idx.append(int(np.random.choice(pool)))
                cat_level_scores[(0, grp)] += recency_w * ACTION_BASE_WEIGHT["category_view"]

        elif action_type == "search":
            sku_list   = value if isinstance(value, list) else [value]
            valid_idxs = [sku2idx[str(s).strip()]
                          for s in sku_list if str(s).strip() in sku2idx]
            if valid_idxs:
                total_w   = recency_w * ACTION_BASE_WEIGHT["category_view"]
                per_sku_w = total_w / len(valid_idxs)
                for idx in valid_idxs:
                    seq_idx.append(idx)
                    finest = _get_finest_cat(sku_meta.get(idx, {}))
                    if finest:
                        level_i, cat_val = finest
                        cat_level_scores[(level_i, cat_val)] += per_sku_w

    if not seq_idx:
        return []

    seq_trunc = seq_idx[-MAX_SEQ_LEN:]
    padded    = [0] * (MAX_SEQ_LEN - len(seq_trunc)) + seq_trunc
    t_in      = torch.tensor([padded], dtype=torch.long)

    with torch.no_grad():
        logits, _ = model(t_in)
    scores = logits[0].cpu().numpy().copy()

    for idx in (viewed_skus | cart_skus):
        scores[idx - 1] = -np.inf

    if cat_level_scores:
        max_acc = max(cat_level_scores.values())
        for (level_i, cat_val), acc_score in cat_level_scores.items():
            boost = (acc_score / max_acc) * category_boost * CAT_LEVEL_FACTORS[level_i]
            for idx in level_cat_to_skuidx.get((level_i, cat_val), []):
                if scores[idx - 1] != -np.inf:
                    scores[idx - 1] += boost

    valid_mask  = scores != -np.inf
    noise_sigma = NOISE_STD * float(np.std(scores[valid_mask])) if valid_mask.any() else 0.0
    if noise_sigma > 0:
        noise              = np.random.normal(0.0, noise_sigma, size=scores.shape)
        noise[~valid_mask] = 0.0
        scores            += noise

    pool_size   = top_k * 25
    top_pool    = np.argsort(scores)[-pool_size:][::-1]
    pool_scores = scores[top_pool]
    shifted     = pool_scores - pool_scores.max()
    probs       = np.exp(shifted / TEMPERATURE)
    probs      /= probs.sum()

    sample_size = min(top_k * 3, pool_size)
    sampled     = np.random.choice(top_pool, size=sample_size, replace=False, p=probs)

    results, seen = [], set()

    for raw_idx in sampled:
        item_idx = raw_idx + 1
        if item_idx not in sku_meta:
            continue
        meta = sku_meta[item_idx]
        if meta["sku"] in seen:
            continue
        if not _passes_filters(meta, stock_filter, promo_only):
            continue
        seen.add(meta["sku"])
        results.append(_build_result(meta, len(results) + 1))
        if len(results) >= top_k:
            break

    if results and EPSILON > 0:
        n_swap         = max(1, round(len(results) * EPSILON))
        swap_positions = np.random.choice(
            len(results), size=min(n_swap, len(results)), replace=False
        ).tolist()
        for pos in swap_positions:
            original_cat = results[pos]["cat_group"]
            pool = [
                idx for idx in cat_group_to_skuidx.get(original_cat, [])
                if idx in sku_meta
                and sku_meta[idx]["sku"] not in seen
                and _passes_filters(sku_meta[idx], stock_filter, promo_only)
            ]
            if not pool:
                continue
            idx  = int(np.random.choice(pool))
            meta = sku_meta[idx]
            seen.add(meta["sku"])
            results[pos] = _build_result(meta, results[pos]["rank"])

    return results


# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    try:
        catalog = _get_active_catalog()
        return jsonify({
            "status"        : "ok",
            "products"      : catalog["product_count"],
            "promo_count"   : catalog["promo_count"],
            "catalog_age_s" : round(time.time() - catalog["loaded_at"]),
            "update_running": _update_state["running"],
        })
    except RuntimeError as e:
        return jsonify({
            "status"        : "no_catalog",
            "error"         : str(e),
            "update_running": _update_state["running"],
            "update_error"  : _update_state["error"],
        }), 503


@app.route("/update", methods=["POST"])
def update_catalog():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Champ 'file' manquant."}), 400

    file = request.files["file"]

    if not file.filename or not file.filename.endswith(".csv"):
        return jsonify({"success": False, "error": "Le fichier doit être un .csv"}), 400

    if not _update_lock.acquire(blocking=False):
        return jsonify({
            "success"   : False,
            "error"     : "Une mise à jour est déjà en cours.",
            "started_at": _update_state.get("started_at"),
        }), 409

    csv_bytes = file.read()

    def _run():
        try:
            _do_update(csv_bytes)
        finally:
            _update_lock.release()

    threading.Thread(target=_run, daemon=True).start()

    return jsonify({
        "success" : True,
        "message" : "Chargement du catalog lancé en background.",
        "filename": file.filename,
    }), 202


@app.route("/recommend", methods=["POST"])
def recommend_route():
    try:
        body    = request.get_json()
        actions = [(a["type"], a["value"]) for a in body.get("actions", [])]
        results = recommend(
            actions,
            top_k          = int(body.get("top_k", 10)),
            category_boost = float(body.get("category_boost", 5.0)),
            stock_filter   = bool(body.get("stock_filter", False)),
            promo_only     = bool(body.get("promo_only", False)),
        )
        return jsonify({"success": True, "count": len(results), "results": results})
    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)