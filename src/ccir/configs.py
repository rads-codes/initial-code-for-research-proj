from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Sequence, Any

'''
Knobs for the following
Languages processed (amount and which ones)
Number of claims evaluated per language
How many articles are looked at initially from SerpAPI per claim (K)
Minimum and maximum content length for articles (Min and Max)
BM25 selects top L articles, choose L
Which corruption settings are used (targeted removed/random removed/random replaced)
Percentage of sentences in each level (i.e. 20%, 40%, 60%)
Models used
Judges used

'''

"""
src/ccir/configs.py

Central configuration ("run knobs") for the CCIR pipeline.

Per project outline, configs.py owns:
- languages processed (which + how many)
- number of claims per language
- SerpAPI K (initial URLs per claim)
- min/max content length for cached plaintext
- BM25 top-L selection
- corruption methods + corruption levels (e.g., 20/40/60%)
- model list + judge list

Downstream expectations (from outline):
- Step 08 uses this to decide which corruption methods + levels to materialize.
- Step 09/10 use model_id/judge_id for output file naming.

Keep this file import-safe (no side effects, no env reads).
"""


# -----------------------------
# Enums
# -----------------------------

class CorruptionMethod(str, Enum):
    RANDOM_DROP = "random_drop"
    TARGETED_DROP = "targeted_drop"
    REPLACEMENT_MIX = "replacement_mix"
    MISLEADING_EDIT = "misleading_edit"


class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    HF_LOCAL = "hf_local"
    OTHER = "other"


# -----------------------------
# Component configs
# -----------------------------

@dataclass(frozen=True)
class DatasetConfig:
    """
    Languages + per-language claim counts.
    - languages: the set/order you want to process.
    - claims_per_language_default: used if a language is not in claims_per_language_override.
    - claims_per_language_override: optional per-language override.
    """
    languages: List[str] = field(default_factory=lambda: ["en"])
    claims_per_language_default: int = 25
    claims_per_language_override: Dict[str, int] = field(default_factory=dict)

    def claims_for_lang(self, lang: str) -> int:
        return int(self.claims_per_language_override.get(lang, self.claims_per_language_default))


@dataclass(frozen=True)
class RetrievalConfig:
    """
    Evidence URL collection + caching constraints.
    - serpapi_k: initial candidate URLs per claim from SerpAPI (K).
    - min_chars / max_chars: content-length bounds for cached plaintext after cleaning.
      (04 drops below min, truncates above max per outline.)
    - fetch_timeout_s / fetch_retries: used by web/fetch.py (even if you wire it later).
    """
    serpapi_k: int = 8
    min_chars: int = 1500
    max_chars: int = 12000
    fetch_timeout_s: int = 20
    fetch_retries: int = 2
    # guardrails for URL collection
    drop_youtube: bool = True
    drop_factcheck_sites: bool = True
    drop_paywalled: bool = True


@dataclass(frozen=True)
class Bm25Config:
    """
    BM25 ranking selection.
    - top_l: number of top documents kept after BM25 (L).
    """
    top_l: int = 3


@dataclass(frozen=True)
class CorruptionConfig:
    """
    Corruption plan for step 08.
    - methods: which corruption strategies to generate.
    - levels: fraction of sentences affected for each method (e.g., [0.2, 0.4, 0.6]).
    - random_seed: seed for any pseudorandom selection (random_drop, replacement_mix).
    - top_k_candidates: for misleading_edit, number of top pool candidates to sample
      replacement from (ranked by embedding similarity to original span).
    """
    methods: List[CorruptionMethod] = field(
        default_factory=lambda: [CorruptionMethod.RANDOM_DROP, CorruptionMethod.TARGETED_DROP, CorruptionMethod.REPLACEMENT_MIX]
    )
    levels: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6])
    random_seed: int = 12345
    top_k_candidates: int = 5

    def variant_names(self) -> List[str]:
        """
        Stable variant name list used for folder naming under:
        runs/<run_id>/cache/corrupted/<variant_name>/<claim_id>/<url_id>
        """
        out: List[str] = []
        for m in self.methods:
            for lvl in self.levels:
                pct = int(round(lvl * 100))
                out.append(f"{m.value}_{pct}")
        return out


@dataclass(frozen=True)
class TranslationConfig:
    """
    Configuration for the Romanian-to-English machine translation condition (ro_mt_en).

    Activated by adding "ro_mt_en" to dataset.languages (e.g. ["en", "de", "ro", "ro_mt_en"]).
    When active, step02b reads Romanian claims from forLLMs.jsonl, translates each to English
    using the specified LLM, and appends new rows with lang="ro_mt_en" and the English
    claim_text. All other fields (claim_id suffix "_mt", claim_date, etc.) are preserved so
    downstream steps treat them as normal rows.

    The translated rows use English-language SerpAPI retrieval (step03), giving English
    evidence for what were originally Romanian claims. Step11 can compare "ro" vs "ro_mt_en"
    on the same source claims by matching claim_ids (ro_mt_en rows have claim_id = orig+"_mt").

    Fields:
    - provider: which API provider to use for translation calls.
    - model_name: exact model name on that provider (e.g., "openai/gpt-4o-mini").
    - temperature: sampling temperature (0.0 = deterministic, recommended).
    - max_tokens: max tokens for the translated output.
    """
    provider: ModelProvider = ModelProvider.OPENROUTER
    model_name: str = "openai/gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512


@dataclass(frozen=True)
class ModelSpec:
    """
    A single fact-checking model used in step 09.
    - model_id: stable identifier for output filenames (avoid spaces).
    - provider/name: how runner.py will resolve the model.
    - temperature/top_p/max_tokens: default decode params.
    """
    model_id: str
    provider: ModelProvider
    name: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512


@dataclass(frozen=True)
class JudgeSpec:
    """
    A single judge model used in step 10.
    - judge_id: stable identifier for output filenames.
    - provider/name: how judge/runner.py will resolve the model.
    """
    judge_id: str
    provider: ModelProvider
    name: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 800


@dataclass(frozen=True)
class RunConfig:
    """
    Top-level run configuration.

    This is what config_loader.load_config() should return/import, and what __main__.py
    should hash into config_hash for lineage (run_metadata.jsonl).
    """
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    bm25: Bm25Config = field(default_factory=Bm25Config)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)

    translation: TranslationConfig = field(default_factory=TranslationConfig)

    models: List[ModelSpec] = field(default_factory=lambda: [
        ModelSpec(model_id="llama3_8b_ollama", provider=ModelProvider.OLLAMA, name="llama3:8b-instruct", temperature=0.0),
    ])
    judges: List[JudgeSpec] = field(default_factory=lambda: [
        JudgeSpec(judge_id="gpt4o_mini_openrouter", provider=ModelProvider.OPENROUTER, name="openai/gpt-4o-mini", temperature=0.0),
    ])

    # Repro / housekeeping
    code_version: Optional[str] = None  # optional manual version string stored in lineage (run_metadata)
    global_seed: int = 12345

    def validate(self) -> None:
        # Dataset
        if not self.dataset.languages:
            raise ValueError("dataset.languages must be a non-empty list (e.g., ['en', 'de']).")
        if len(set(self.dataset.languages)) != len(self.dataset.languages):
            raise ValueError(f"dataset.languages has duplicates: {self.dataset.languages}")
        if self.dataset.claims_per_language_default <= 0:
            raise ValueError("dataset.claims_per_language_default must be > 0.")
        for lang, n in self.dataset.claims_per_language_override.items():
            if not lang or not isinstance(lang, str):
                raise ValueError(f"Invalid language key in claims_per_language_override: {lang!r}")
            if n <= 0:
                raise ValueError(f"claims_per_language_override[{lang}] must be > 0; got {n}")

        # Retrieval
        if self.retrieval.serpapi_k <= 0:
            raise ValueError("retrieval.serpapi_k (K) must be > 0.")
        if self.retrieval.min_chars < 0 or self.retrieval.max_chars <= 0:
            raise ValueError("retrieval.min_chars must be >= 0 and retrieval.max_chars must be > 0.")
        if self.retrieval.min_chars > self.retrieval.max_chars:
            raise ValueError("retrieval.min_chars cannot exceed retrieval.max_chars.")
        if self.retrieval.fetch_timeout_s <= 0:
            raise ValueError("retrieval.fetch_timeout_s must be > 0.")
        if self.retrieval.fetch_retries < 0:
            raise ValueError("retrieval.fetch_retries must be >= 0.")

        # BM25
        if self.bm25.top_l <= 0:
            raise ValueError("bm25.top_l (L) must be > 0.")
        if self.bm25.top_l > self.retrieval.serpapi_k:
            raise ValueError(f"bm25.top_l (L={self.bm25.top_l}) cannot exceed retrieval.serpapi_k (K={self.retrieval.serpapi_k}).")

        # Corruption
        if not self.corruption.methods:
            raise ValueError("corruption.methods must be non-empty.")
        if not self.corruption.levels:
            raise ValueError("corruption.levels must be non-empty (e.g., [0.2, 0.4, 0.6]).")
        for lvl in self.corruption.levels:
            if not (0.0 < float(lvl) < 1.0):
                raise ValueError(f"corruption.levels entries must be in (0,1); got {lvl}")
        if len(set(self.corruption.levels)) != len(self.corruption.levels):
            raise ValueError(f"corruption.levels has duplicates: {self.corruption.levels}")
        if self.corruption.top_k_candidates < 1:
            raise ValueError("corruption.top_k_candidates must be >= 1.")

        # Models/Judges
        if not self.models:
            raise ValueError("models must be non-empty (at least one ModelSpec).")
        if len({m.model_id for m in self.models}) != len(self.models):
            raise ValueError("models has duplicate model_id values.")
        for m in self.models:
            if not m.model_id or " " in m.model_id:
                raise ValueError(f"ModelSpec.model_id must be non-empty and contain no spaces; got {m.model_id!r}")
            if m.max_tokens <= 0:
                raise ValueError(f"ModelSpec.max_tokens must be > 0 for {m.model_id}")
            if not (0.0 <= m.temperature <= 2.0):
                raise ValueError(f"ModelSpec.temperature out of range for {m.model_id}: {m.temperature}")
            if not (0.0 < m.top_p <= 1.0):
                raise ValueError(f"ModelSpec.top_p out of range for {m.model_id}: {m.top_p}")

        if not self.judges:
            raise ValueError("judges must be non-empty (at least one JudgeSpec).")
        if len({j.judge_id for j in self.judges}) != len(self.judges):
            raise ValueError("judges has duplicate judge_id values.")
        for j in self.judges:
            if not j.judge_id or " " in j.judge_id:
                raise ValueError(f"JudgeSpec.judge_id must be non-empty and contain no spaces; got {j.judge_id!r}")
            if j.max_tokens <= 0:
                raise ValueError(f"JudgeSpec.max_tokens must be > 0 for {j.judge_id}")
            if not (0.0 <= j.temperature <= 2.0):
                raise ValueError(f"JudgeSpec.temperature out of range for {j.judge_id}: {j.temperature}")
            if not (0.0 < j.top_p <= 1.0):
                raise ValueError(f"JudgeSpec.top_p out of range for {j.judge_id}: {j.top_p}")

        # Translation
        t = self.translation
        if not (0.0 <= t.temperature <= 2.0):
            raise ValueError(f"translation.temperature must be in [0.0, 2.0]; got {t.temperature}")
        if t.max_tokens <= 0:
            raise ValueError(f"translation.max_tokens must be > 0; got {t.max_tokens}")
        if not t.model_name:
            raise ValueError("translation.model_name must be non-empty")
        if "ro_mt_en" in self.dataset.languages and "ro" not in self.dataset.languages:
            raise ValueError("dataset.languages includes 'ro_mt_en' but not 'ro'; 'ro' is required as the translation source.")

        if self.global_seed < 0:
            raise ValueError("global_seed must be >= 0.")
        if self.corruption.random_seed < 0:
            raise ValueError("corruption.random_seed must be >= 0.")

    def to_dict(self) -> Dict[str, Any]:
        """
        Deterministic dict for hashing/serialization.
        Note: keep list ordering stable where it matters (languages, models, judges).
        """
        d = asdict(self)

        # Normalize enums to values
        d["corruption"]["methods"] = [m.value for m in self.corruption.methods]
        d["translation"]["provider"] = self.translation.provider.value

        # Ensure stable ordering for dicts that don't have semantic order
        d["dataset"]["claims_per_language_override"] = dict(sorted(d["dataset"]["claims_per_language_override"].items(), key=lambda kv: kv[0]))

        return d


# A single default config object imported by config_loader / __main__ / scripts.
#DEFAULT_CONFIG = RunConfig()
# Optional: validate on import to fail fast during development; comment out if you prefer.
#DEFAULT_CONFIG.validate()

# -----------------------------
# RUN SETTINGS (EDIT THIS PER RUN)
# Only edit values in this block.
# Everything above defines the config schema and validation rules.
# -----------------------------

DEFAULT_CONFIG = RunConfig(

    # -------------------------
    # DATASET SETTINGS
    # -------------------------
    dataset=DatasetConfig(

        # List[str]
        # Languages to evaluate.
        # Use ISO-like language codes (e.g., "en", "de", "ro", "el").
        # Example formats:
        # ["en"]
        # ["en", "de"]
        # ["en", "de", "ro", "el"]
        languages=["en", "de", "el", "ro", "ro_mt_en"],

        # int
        # Default number of claims evaluated for each language.
        # Must be > 0.
        # Example formats:
        # 10
        # 25
        # 100
        claims_per_language_default=190,

        # Dict[str, int]
        # Optional override for specific languages.
        # Key = language code
        # Value = number of claims for that language.
        #
        # Example formats:
        # {}
        # {"en": 50}
        # {"en": 50, "de": 25}
        claims_per_language_override={},  # optional

    ),

    # -------------------------
    # RETRIEVAL SETTINGS
    # -------------------------
    retrieval=RetrievalConfig(

        # int
        # K = number of URLs retrieved from SerpAPI per claim
        # before filtering and ranking.
        #
        # Example formats:
        # 5
        # 8
        # 10
        serpapi_k=8,

        # int
        # Minimum article length (in characters) after cleaning.
        # Articles shorter than this will be dropped.
        #
        # Example formats:
        # 800
        # 1200
        # 1500
        min_chars=800,

        # int
        # Maximum article length (in characters).
        # Articles longer than this will be truncated.
        #
        # Example formats:
        # 8000
        # 10000
        # 15000
        max_chars=10000,

        # int
        # Timeout for webpage fetch requests in seconds.
        #
        # Example formats:
        # 10
        # 20
        # 30
        fetch_timeout_s=25,

        # int
        # Number of retries if webpage fetch fails.
        #
        # Example formats:
        # 0
        # 1
        # 2
        fetch_retries=1,

        # bool
        # If True, remove YouTube URLs from search results.
        drop_youtube=True,

        # bool
        # If True, remove known fact-checking websites.
        # (Prevents models from simply reading fact-checks.)
        drop_factcheck_sites=True,

        # bool
        # If True, remove paywalled pages that cannot be scraped.
        drop_paywalled=True,
    ),

    # -------------------------
    # BM25 RANKING SETTINGS
    # -------------------------
    bm25=Bm25Config(

        # int
        # L = number of top articles selected after BM25 ranking.
        #
        # IMPORTANT:
        # Must satisfy: L <= serpapi_k
        #
        # Example formats:
        # 2
        # 3
        # 5
        top_l=3,
    ),

    # -------------------------
    # CORRUPTION SETTINGS
    # -------------------------
    corruption=CorruptionConfig(

        # List[CorruptionMethod]
        # Which corruption strategies to generate.
        #
        # Available values:
        # CorruptionMethod.RANDOM_DROP
        # CorruptionMethod.TARGETED_DROP
        # CorruptionMethod.REPLACEMENT_MIX
        # CorruptionMethod.MISLEADING_EDIT
        #
        # Example formats:
        # [CorruptionMethod.RANDOM_DROP]
        # [CorruptionMethod.RANDOM_DROP, CorruptionMethod.TARGETED_DROP]
        # [CorruptionMethod.RANDOM_DROP, CorruptionMethod.TARGETED_DROP, CorruptionMethod.REPLACEMENT_MIX]
        methods=[
            CorruptionMethod.RANDOM_DROP,
            CorruptionMethod.TARGETED_DROP,
            CorruptionMethod.REPLACEMENT_MIX,
            CorruptionMethod.MISLEADING_EDIT,
        ],

        # int
        # For misleading_edit: number of top pool candidates to sample
        # replacement entity from (ranked by embedding similarity to original span).
        #
        # Example formats:
        # 3
        # 5
        # 10
        top_k_candidates=3,

        # List[float]
        # Fraction of sentences corrupted.
        #
        # Must be values between 0 and 1 (exclusive).
        #
        # Example formats:
        # [0.2]
        # [0.2, 0.4]
        # [0.2, 0.4, 0.6]
        levels=[0.3, 0.6],

        # int
        # Random seed controlling which sentences are corrupted.
        # Change this if you want a different corruption sample.
        #
        # Example formats:
        # 123
        # 12345
        random_seed=12345,
    ),

    # -------------------------
    # TRANSLATION SETTINGS (STEP 02b)
    # -------------------------
    translation=TranslationConfig(

        # ModelProvider enum
        # Provider used for translation LLM calls.
        # Only used when "ro_mt_en" is in dataset.languages.
        # Options: ModelProvider.OPENROUTER, ModelProvider.OPENAI, ModelProvider.OLLAMA
        provider=ModelProvider.OPENROUTER,

        # str
        # Exact model name on the provider for translation.
        # A fast, capable model is recommended (translation is straightforward).
        # Example: "openai/gpt-4o-mini", "openai/gpt-4o", "google/gemini-flash-1.5"
        model_name="openai/gpt-4o-mini",

        # float
        # Temperature for translation (0.0 = deterministic, recommended for reproducibility).
        temperature=0.0,

        # int
        # Max tokens for the translated output. Claims are short, 512 is ample.
        max_tokens=512,
    ),

    # -------------------------
    # MODEL SETTINGS (STEP 09)
    # -------------------------
    models=[

        ModelSpec(

            # str
            # Unique identifier used in filenames.
            # Must contain no spaces.
            #
            # Example formats:
            # "llama3_8b_ollama"
            # "deepseek_r1_openrouter"
            model_id="llama8B",

            # ModelProvider enum
            # Where the model is hosted.
            #
            # Options:
            # ModelProvider.OLLAMA
            # ModelProvider.OPENROUTER
            # ModelProvider.OPENAI
            # ModelProvider.HF_LOCAL
            provider=ModelProvider.OPENROUTER,

            # str
            # Exact model name used by the provider.
            #
            # Example formats:
            # "llama3:8b-instruct"
            # "deepseek/deepseek-r1"
            name="meta-llama/llama-3.1-8b-instruct",

            # float
            # Sampling temperature.
            #
            # Example formats:
            # 0.0
            # 0.2
            temperature=0.0,

            # int
            # Maximum tokens generated for the response.
            #
            # Example formats:
            # 256
            # 512
            # 1024
            max_tokens=5000,
        ),
ModelSpec(

            # str
            # Unique identifier used in filenames.
            # Must contain no spaces.
            #
            # Example formats:
            # "llama3_8b_ollama"
            # "deepseek_r1_openrouter"
            model_id="qwen7B",

            # ModelProvider enum
            # Where the model is hosted.
            #
            # Options:
            # ModelProvider.OLLAMA
            # ModelProvider.OPENROUTER
            # ModelProvider.OPENAI
            # ModelProvider.HF_LOCAL
            provider=ModelProvider.OPENROUTER,

            # str
            # Exact model name used by the provider.
            #
            # Example formats:
            # "llama3:8b-instruct"
            # "deepseek/deepseek-r1"
            name="qwen/qwen-2.5-7b-instruct",

            # float
            # Sampling temperature.
            #
            # Example formats:
            # 0.0
            # 0.2
            temperature=0.0,

            # int
            # Maximum tokens generated for the response.
            #
            # Example formats:
            # 256
            # 512
            # 1024
            max_tokens=5000,
        ),
ModelSpec(

            # str
            # Unique identifier used in filenames.
            # Must contain no spaces.
            #
            # Example formats:
            # "llama3_8b_ollama"
            # "deepseek_r1_openrouter"
            model_id="gemma9B",

            # ModelProvider enum
            # Where the model is hosted.
            #
            # Options:
            # ModelProvider.OLLAMA
            # ModelProvider.OPENROUTER
            # ModelProvider.OPENAI
            # ModelProvider.HF_LOCAL
            provider=ModelProvider.OPENROUTER,

            # str
            # Exact model name used by the provider.
            #
            # Example formats:
            # "llama3:8b-instruct"
            # "deepseek/deepseek-r1"
            name="google/gemma-2-9b-it",

            # float
            # Sampling temperature.
            #
            # Example formats:
            # 0.0
            # 0.2
            temperature=0.0,

            # int
            # Maximum tokens generated for the response.
            #
            # Example formats:
            # 256
            # 512
            # 1024
            max_tokens=5000,
        ),
    ],

    # -------------------------
    # JUDGE MODEL SETTINGS (STEP 10)
    # -------------------------
    judges=[

        JudgeSpec(

            # str
            # Unique identifier for judge output filenames.
            #
            # Example formats:
            # "gpt4o_mini_openrouter"
            # "gpt4_turbo_openai"
            judge_id="deepseekR1",

            # ModelProvider enum
            provider=ModelProvider.OPENROUTER,

            # str
            # Actual judge model name.
            #
            # Example formats:
            # "openai/gpt-4o-mini"
            # "openai/gpt-4-turbo"
            name="deepseek/deepseek-r1",

            # float
            # Judge temperature (usually 0 for deterministic scoring)
            temperature=0.0,

            # int
            # Maximum tokens allowed in judge output.
            max_tokens=10000,
        ),
JudgeSpec(

            # str
            # Unique identifier for judge output filenames.
            #
            # Example formats:
            # "gpt4o_mini_openrouter"
            # "gpt4_turbo_openai"
            judge_id="gpt-4.1",

            # ModelProvider enum
            provider=ModelProvider.OPENROUTER,

            # str
            # Actual judge model name.
            #
            # Example formats:
            # "openai/gpt-4o-mini"
            # "openai/gpt-4-turbo"
            name="openai/gpt-4.1",

            # float
            # Judge temperature (usually 0 for deterministic scoring)
            temperature=0.0,

            # int
            # Maximum tokens allowed in judge output.
            max_tokens=10000,
        ),
    ],

    # -------------------------
    # RUN METADATA
    # -------------------------

    # str or None
    # Optional manual version label for the experiment.
    # Change this whenever you change settings.
    #
    # Example formats:
    # "pilot_run"
    # "multilingual_v1"
    # "experiment_A"
    code_version="full_run",

    # int
    # Global random seed used for reproducibility.
    #
    # Example formats:
    # 42
    # 123
    # 12345
    global_seed=12345,
)

# Fail fast if any config rule is violated
# (e.g., L > K, invalid corruption levels, duplicate model IDs).
DEFAULT_CONFIG.validate()