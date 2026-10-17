#!/usr/bin/env python3
"""Original pixel-to-head image training plan, using preserved historical recipes.

Pretrained public backbones are inputs, not randomly initialized. No detector
checkpoint is used to initialize detector-head training except earlier stages
of this same plan. V696's optional student stage needs its preserved teacher.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def source_fingerprint(source):
    """Invalidate resumed stages when recipes or their imported code change."""
    digest = hashlib.sha256()
    for base in (source, HERE):
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".json"}:
                digest.update(str(path.relative_to(base)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def can_resume(marker, stage, fingerprint, work):
    if not marker.is_file():
        return False
    saved = json.loads(marker.read_text())
    if saved.get("source_fingerprint") != fingerprint or saved.get("command") != stage["command"]:
        raise RuntimeError(f"Recipe changed since {stage['name']} completed; use a new work directory")
    if not all((work / q).is_file() for q in stage["outputs"]):
        raise RuntimeError(f"Completed stage {stage['name']} has missing outputs; use a new work directory")
    return True


def plan(source: Path, work: Path):
    result = []
    def add(name, script, config=None, extra=(), outputs=()):
        command = [sys.executable, str(source / "scripts" / script), "--project-root", str(work)]
        if config:
            command += ["--config", str(source / "configs" / config)]
        command += list(extra)
        result.append({"name": name, "command": command, "outputs": list(outputs)})
    for version, stem, paired in [
        ("V21", "clip_adapter_external", False), ("V28", "clip_adapter_modern_external", False),
        ("V29", "clip_adapter_multidepth_patch_external", False),
        ("V31", "clip_adapter_paired_external", True), ("V36", "clip_adapter_paired_grid", True),
    ]:
        add(version + "-external", "pretrain_clip_adapter_paired.py" if paired else "pretrain_clip_adapter_external.py",
            f"{version}_{stem}_pretrain.yaml", outputs=[f"outputs/{version}_{stem}/pretrain/external_pretrain.pth"])
    for v, stem in [("V29", "clip_adapter_multidepth_patch_external"), ("V36", "clip_adapter_paired_grid")]:
        add(v + "-full", "build_image_submission_v1.py", f"{v}_{stem}_full_e7.yaml", outputs=[f"outputs/{v}_{stem}/full_e7/trainable_e007.pth"])
    for v, stem, epochs, script, weight in [
        ("V47", "pe_core_g14_linear", 100, "build_pe_core_submission.py", "linear_head.pth"),
        ("V51", "pe_core_aigi_breadth_linear", 22, "build_pe_core_submission.py", "linear_head.pth"),
        ("V54", "pe_core_residual_mlp", 3, "build_pe_residual_submission.py", "residual_mlp.pth"),
        ("V56", "pe_core_openfake_residual_mlp", 3, "build_pe_residual_submission.py", "residual_mlp.pth"),
        ("V58", "pe_core_aigi_residual_mlp_seed2", 3, "build_pe_residual_submission.py", "residual_mlp.pth"),
        ("V60", "cross_backbone_residual", 3, "build_cross_backbone_residual_submission.py", "clip_residual.pth"),
    ]:
        add(v + "-full", script, f"{v}_{stem}_full_e{epochs}.yaml", outputs=[f"outputs/{v}_{stem}/full_e{epochs}/{weight}"])
    for branch, config, folder in [
        ("core", "V66_pe_core_multidepth_residual_fold0.yaml", "V66_pe_core_multidepth_residual"),
        ("spatial", "V76_pe_spatial_multidepth_extract.yaml", "V76_pe_spatial_multidepth_classifier"),
    ]:
        add(branch + "-statistics", "train_pe_core_multidepth_residual.py", config, ["--extract-only"],
            [f"outputs/{folder}/features/competition_multidepth_statistics.npz", f"outputs/{folder}/features/aigi_multidepth_statistics.npz"])
        add(branch + "-test-statistics", "extract_pe_multidepth_test.py", f"PE_{branch}_multidepth_test_extract.yaml",
            outputs=[f"outputs/{folder}/features/test_multidepth_statistics.npz"])
    for v, stem in [("V67", "pe_core_multidepth_classifier"), ("V71", "pe_core_multidepth_pairwise"),
                    ("V76", "pe_spatial_multidepth_classifier"), ("V77", "pe_spatial_multidepth_pairwise")]:
        add(v + "-full", "build_pe_multidepth_submission.py", f"{v}_{stem}_full_e10.yaml", outputs=[f"outputs/{v}_{stem}/full_e10/multidepth_classifier.pth"])
    add("V86-full", "build_dual_pe_fusion_submission.py", "V86_dual_pe_multidepth_fusion_full_e5.yaml", outputs=["outputs/V86_dual_pe_multidepth_fusion/full_e5/dual_pe_fusion.pth"])
    add("V90-ensemble", "build_weighted_rank_submission.py", "V90_image_weighted_rank_ensemble_full.yaml", outputs=["outputs/submissions/V90_image_weighted_rank_ensemble.csv"])
    for mode in ["competition", "test"]:
        for name, script, folder in [
            ("gray", "extract_pe_core_grayscale_features.py", "V161_pe_core_grayscale_rbf"),
            ("views", "extract_pe_core_multiview_features.py", "V118_pe_core_multiview_rbf"),
            ("face", "extract_face_crop_vit_features.py", "V107_face_crop_rbf"),
        ]:
            out = f"outputs/{folder}/features/{mode}.npz"
            add(f"{name}-{mode}", script, None if name == "face" else "V50_pe_core_multiview_fold0.yaml", ["--mode", mode, "--output", out], [out])
    for model, out in [("rine", "V111_rine_ldm_rbf/features/competition_center_features.npz"), ("npr", "V113_npr_rbf/features/competition_full_features.npz")]:
        result.append({"name": model + "-train", "command": [sys.executable, str(HERE / "extract_forensic_train.py"), "--source-root", str(source), "--project-root", str(work), "--model", model], "outputs": ["outputs/" + out]})
    add("bombek-train", "evaluate_bombek_dual.py", extra=["--all-rows", "--feature-cache", "outputs/V99_bombek_dual_residual/features/competition_features.npz"], outputs=["outputs/V99_bombek_dual_residual/features/competition_features.npz"])
    add("forensic-test", "extract_V178_test_forensics.py", outputs=["outputs/V99_bombek_dual_residual/features/test_features.npz", "outputs/V111_rine_ldm_rbf/features/test_center_features.npz", "outputs/V113_npr_rbf/features/test_full_features.npz"])
    add("V178-fit", "train_infer_V178_full.py", outputs=["outputs/V178_full/test_component_predictions.csv", "outputs/V178_full/test_geometry.csv", "outputs/V178_full/models/V178_rine_npr_square_meta_full.joblib"])
    result.append({"name": "prepare-runtime-assets", "command": [sys.executable, str(HERE / "prepare_assets.py"), "--source-root", str(source),
        "--project-root", str(work), "--output", str(work / "assets")], "outputs": ["assets/manifest.json"]})
    return result


def prerequisites(work):
    paths = [
        "data/competition/image/image_sample_data.csv", "data/competition/image/image_test.csv",
        "data/competition/image/image_submission_example.csv", "data/folds/image_folds_v1.csv",
        "data/external/community_forensics_transfer_v1/manifest.csv",
        "data/external/external_transfer_v2_manifest.csv", "data/external/external_transfer_v3_manifest.csv",
        "data/external/external_transfer_v4_manifest.csv", "data/external/aigi_v2_breadth_pairs_manifest.csv",
        "data/external/openfake_frontier_api_manifest.csv",
        ".cache/clip/ViT-L-14-336px.pt", ".cache/clip/ViT-L-14.pt",
        ".cache/pe_core_g14/model.safetensors", ".cache/pe_spatial_g14/model.safetensors",
        ".cache/rine/model_ldm_trainable.pth", ".cache/npr/model_epoch_last_3090.pth",
        ".cache/bombek_dual/pytorch_model.pt", ".cache/deepfake_vit/model.safetensors",
        ".cache/deepfake_vit/config.json",
    ]
    missing = [str(work / p) for p in paths if not (work / p).is_file()]
    for name, sub, expected in [("image_sample_data", "image_sample_data", 1000), ("image_test", "image_test", 20000)]:
        manifest = work / f"data/competition/image/{name}.csv"
        if manifest.is_file():
            with manifest.open() as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != expected or len({r["id"] for r in rows}) != expected:
                missing.append(f"INVALID_ROWS:{manifest} (expected {expected})")
            missing.extend(str(work / f"data/competition/image/data/{sub}" / r["id"]) for r in rows if not (work / f"data/competition/image/data/{sub}" / r["id"]).is_file())
    for rel in paths:
        if not rel.startswith("data/external/") or not (work / rel).is_file():
            continue
        with (work / rel).open() as stream:
            rows = list(csv.DictReader(stream))
        base = work / ("data/external/community_forensics_transfer_v1" if "community_forensics_transfer_v1" in rel else "data/external")
        for row in rows:
            path = row.get("relative_path") or row.get("path")
            if path and not (base / path).is_file():
                missing.append(str(base / path))
    return list(dict.fromkeys(missing))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=ROOT / "research/image")
    p.add_argument("--work-dir", type=Path, default=ROOT / "runs/image_training")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip only stages with successful completion marker and all declared outputs")
    p.add_argument("--through", help="Stop after this named stage")
    p.add_argument("--teacher", type=Path, help="Optional preserved V696 teacher CSV for the SAME 20k input IDs; fit a new deployment student after visual training")
    a = p.parse_args()
    source, work = a.source_root.resolve(), a.work_dir.resolve()
    stages = plan(source, work)
    if a.teacher:
        stages.append({"name": "V696-student", "command": [sys.executable, str(HERE / "refit_student.py"),
            "--components", str(work / "outputs/V178_full/test_component_predictions.csv"),
            "--geometry", str(work / "outputs/V178_full/test_geometry.csv"),
            "--teacher", str(a.teacher.resolve()), "--output", str(work / "assets/distill"),
            "--no-require-reference-parity"], "outputs": ["assets/distill/v696_distill.joblib", "assets/distill/report.json"]})
    if a.through:
        names = [s["name"] for s in stages]
        if a.through not in names:
            p.error("Unknown stage: " + a.through)
        stages = stages[:names.index(a.through)+1]
    if a.plan:
        for stage in stages:
            print(stage["name"] + "\n  " + shlex.join(stage["command"]))
        return
    missing = prerequisites(work)
    if not (source / "docker/v178_image/prepare_assets.py").is_file():
        missing.append(str(source / "docker/v178_image/prepare_assets.py"))
    if a.teacher and not a.teacher.is_file():
        missing.append(str(a.teacher))
    for stage in stages:
        command = stage["command"]
        if not Path(command[1]).is_file():
            missing.append(command[1])
        if "--config" in command and not Path(command[command.index("--config")+1]).is_file():
            missing.append(command[command.index("--config")+1])
    if missing:
        print(json.dumps({"ready": False, "missing_count": len(missing), "missing_first_50": missing[:50]}, indent=2))
        raise SystemExit(2)
    if a.check:
        print(json.dumps({"ready": True, "stages": len(stages), "training_executed": False}))
        return
    protected = [ROOT, ROOT / ".local", ROOT / "research", ROOT / "outputs", ROOT / "docker"]
    if work == ROOT or any(work == q or work.is_relative_to(q) for q in protected[1:]):
        raise ValueError("Training writes must use a dedicated directory, never preserved assets, legacy inputs, source, or the repository root")
    if any((work / q).is_symlink() for q in ["outputs", "assets", "stage_logs"]):
        raise ValueError("Training outputs must not be symlinks into existing assets")
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(work)
    env["PYTHONPATH"] = os.pathsep.join([str(source / "src"), str(source / "scripts"), env.get("PYTHONPATH", "")])
    logdir = work / "stage_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint(source)
    for stage in stages:
        marker = logdir / (stage["name"] + ".done.json")
        if a.resume and can_resume(marker, stage, fingerprint, work):
            continue
        if marker.is_file() and not a.resume:
            raise RuntimeError(f"Existing completed stage {stage['name']}; use a new work directory or --resume")
        print("Running " + stage["name"], flush=True)
        with (logdir / (stage["name"] + ".log")).open("w") as log:
            subprocess.run(stage["command"], cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        absent = [q for q in stage["outputs"] if not (work / q).is_file()]
        if absent:
            raise RuntimeError(f"Stage {stage['name']} finished without declared outputs: {absent}")
        marker.write_text(json.dumps(dict(stage, source_fingerprint=fingerprint), indent=2) + "\n")


if __name__ == "__main__":
    main()
