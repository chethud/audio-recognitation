"""
Standalone inference worker - runs in subprocess. Crashes/OOM here won't kill the API.
Writes JSON result to output file: {"ok": true, "answer": "..."} or {"ok": false, "error": "..."}
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: inference_worker.py <audio_path> <output_json_path> [--question-file <path>]\n")
        sys.exit(1)

    args = sys.argv[1:]
    if "--question-file" in args:
        i = args.index("--question-file")
        if i + 1 < len(args):
            question = Path(args[i + 1]).read_text(encoding="utf-8").strip()
        else:
            question = "What can be inferred from the audio?"
        # Remove --question-file and its arg
        args = [a for j, a in enumerate(args) if j != i and j != i + 1]
    else:
        question = args[1] if len(args) > 2 else "What can be inferred from the audio?"
        args = args[:2]  # audio_path, output_path

    audio_path = args[0]
    output_path = args[1] if len(args) > 1 else None
    if not output_path:
        sys.stderr.write("Missing output path\n")
        sys.exit(1)

    def write_out(obj):
        Path(output_path).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

    if not Path(audio_path).exists():
        write_out({"ok": False, "error": f"File not found: {audio_path}"})
        sys.exit(1)

    try:
        import torch
        import yaml
        from src.models import ALMModel
        from src.utils import load_audio_from_file

        config_path = BASE / "config.yaml"
        checkpoint_path = BASE / "outputs" / "alm_checkpoint.pt"
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ALMModel(
            audio_encoder_name=cfg["audio_encoder"]["model_name"],
            llm_name=cfg["llm"]["model_name"],
            audio_feat_dim=768,
            llm_hidden_size=896,
            num_audio_tokens=32,
            freeze_audio=True,
        )
        if checkpoint_path.exists():
            model.load_state_dict(
                torch.load(str(checkpoint_path), map_location="cpu", weights_only=False),
                strict=False,
            )
        model = model.to(device)
        model.eval()

        audio = load_audio_from_file(
            audio_path,
            sr=cfg["data"]["sample_rate"],
            max_sec=cfg["data"]["max_audio_length_sec"],
        ).to(device)

        llm_cfg = cfg.get("llm", {})
        with torch.no_grad():
            answer = model.generate(
                audio,
                question,
                max_new_tokens=llm_cfg.get("max_new_tokens", 128),
                repetition_penalty=llm_cfg.get("repetition_penalty", 1.2),
                no_repeat_ngram_size=llm_cfg.get("no_repeat_ngram_size", 4),
            )

        write_out({"ok": True, "answer": answer})
    except Exception as e:
        write_out({"ok": False, "error": f"{type(e).__name__}: {str(e)}"})
        sys.exit(1)


if __name__ == "__main__":
    main()
