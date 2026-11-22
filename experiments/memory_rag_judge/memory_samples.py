"""Build sleeper-memory attack samples with the benchmark's own functions (dataset, attack, prompts)."""
import os, pathlib, sys, types
BENCH = pathlib.Path(__file__).resolve().parent  # placeholder, set by caller
def load(root, dataset_file, n, attack="universal_v2_optimized_with_markers", provider="generic", offset=0):
    bench = pathlib.Path(root) / "attacks-bench/memory/LLM-agent-memory-poisoning"
    sys.path.insert(0, str(bench))
    # Office/RTF readers are only needed for .xlsx/.xls/.odt/.rtf documents; the sampled documents are text.
    for name in ("openpyxl", "xlrd", "odf", "odf.opendocument", "odf.text", "odf.teletype", "striprtf", "striprtf.striprtf"):
        try:
            __import__(name)
        except ImportError:
            stub = types.ModuleType(name); stub.__getattr__ = lambda attr: None
            sys.modules[name] = stub
    import importlib
    from sleeper_eval.dataset import load_dataset, SampleMeta
    from sleeper_eval.defenses import defense_config_from_names
    from sleeper_eval.provider_config import get_provider_config
    from sleeper_eval.solver import build_system_prompt, build_user_message
    attack_fn = importlib.import_module(f"sleeper_eval.attacks.{attack}").apply
    # app-side defense of the benchmark itself, e.g. MEM_APP_DEFENSE=untrusted_content_markers (document wrapped in
    # BEGIN_UNTRUSTED_DOCUMENT ... END_UNTRUSTED_DOCUMENT plus a reminder); empty = none
    defense = defense_config_from_names(os.getenv("MEM_APP_DEFENSE", ""))
    pc = get_provider_config(provider, memory_backend_name="tool")
    out = []
    for sample in list(load_dataset(dataset_file))[offset:offset + n]:
        meta = SampleMeta.model_validate(sample.metadata)
        doc = attack_fn(meta.doc_text, meta.goal_text, sample.metadata)
        out.append({"id": f"{pathlib.Path(dataset_file).stem[:5]}_{meta.goal_id}", "goal": meta.goal_text,
                    "query": meta.query_text, "memories": meta.memories,
                    "system": build_system_prompt(meta, defense, provider_config=pc, memory_backend_name="tool"),
                    "user": build_user_message(meta=meta, provider_config=pc, query_text=meta.query_text, document_text=doc, defense_config=defense)})
    return out
