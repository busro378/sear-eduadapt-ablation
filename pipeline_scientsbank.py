#!/usr/bin/env python3
"""
Pipeline lengkap SEAR Paper 1 di komputer sendiri (tanpa Colab, tanpa file awal):
  1. Instal paket yang belum ada
  2. Unduh SciEntsBank (SemEval-2013 Task 7) -> korpus C1-C4
  3. Latih SBERT-EduAdapt hanya dari split train (CPU bisa 30-90 menit; dilewati jika sudah ada)
  4. Ablation TF-IDF vs General SBERT vs SBERT-EduAdapt: semua test + per split
  5. Kemas hasil ke sear_results.zip
Jalankan dengan klik ganda PIPELINE_SCIENTSBANK.bat. Log: log_pipeline.txt
Aman dijalankan ulang: langkah yang sudah selesai dilewati.
"""
import os
import shutil
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
LOG = os.path.join(HERE, "log_pipeline.txt")
_log = open(LOG, "a", encoding="utf-8")

CORPUS = os.path.join(HERE, "scientsbank_C1C4.csv")
PAIRS = os.path.join(HERE, "SBERT_training_pairs.csv")
MODEL = os.path.join(HERE, "SBERT-EduAdapt-v1-heldout")
OUT = os.path.join(HERE, "sear_results")


def say(msg=""):
    print(msg, flush=True)
    _log.write(msg + "\n")
    _log.flush()


def jalankan(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", cwd=HERE)
    for line in p.stdout:
        line = line.rstrip("\n")
        print("   " + line, flush=True)
        if "it/s]" not in line and "s/it]" not in line:   # progress bar tidak dicatat ke log
            _log.write("   " + line + "\n")
    p.wait()
    return p.returncode


def langkah1_paket():
    perlu = {"pandas": "pandas", "numpy": "numpy", "scipy": "scipy", "sklearn": "scikit-learn",
             "matplotlib": "matplotlib", "datasets": "datasets", "accelerate": "accelerate",
             "sentence_transformers": "sentence-transformers"}
    kurang = []
    for mod, pkg in perlu.items():
        try:
            __import__(mod)
        except ImportError:
            kurang.append(pkg)
    if not kurang:
        say("[1/5] Semua paket sudah terpasang.")
        return
    say(f"[1/5] Menginstal: {', '.join(kurang)}  (pertama kali 5-15 menit, butuh internet)")
    if jalankan([sys.executable, "-m", "pip", "install", *kurang]) != 0:
        raise RuntimeError("Instalasi paket gagal. Lihat pesan pip di atas.")


def langkah2_korpus():
    import pandas as pd
    if os.path.exists(CORPUS):
        say(f"[2/5] Korpus sudah ada: {CORPUS}")
        corpus = pd.read_csv(CORPUS)
    else:
        say("[2/5] Mengunduh SciEntsBank dari Hugging Face...")
        from datasets import ClassLabel, load_dataset
        sys.path.insert(0, HERE)
        from run_ablation import normalize_label
        ds = load_dataset("nkazi/SciEntsBank")
        say(str(ds))

        def pick(cols, *keys):
            for k in keys:
                for c in cols:
                    if k in c.lower():
                        return c
            raise KeyError(f"Kolom {keys} tidak ditemukan di {cols}")

        frames = []
        for split, d in ds.items():
            cols = d.column_names
            ref, stud, lab = pick(cols, "reference"), pick(cols, "student"), pick(cols, "label")
            q = next((c for c in cols if "question" in c.lower()), None)
            f = d.features[lab]
            labels = ([f.int2str(x) for x in d[lab]] if isinstance(f, ClassLabel)
                      else [str(x) for x in d[lab]])
            frames.append(pd.DataFrame({"question": d[q] if q else "",
                                        "reference_answer": d[ref], "student_answer": d[stud],
                                        "original_label": labels, "split": split}))
        corpus = pd.concat(frames, ignore_index=True)
        corpus["normalized_label"] = corpus["original_label"].map(normalize_label)
        corpus.to_csv(CORPUS, index=False)
        say("      Label asli -> C1-C4:")
        say(pd.crosstab(corpus["original_label"], corpus["normalized_label"]).to_string())
    say("      Jumlah per split x label:")
    say(pd.crosstab(corpus["split"], corpus["normalized_label"], margins=True).to_string())
    tests = [s for s in corpus["split"].astype(str).unique() if "test" in s.lower()]
    if not tests:
        raise RuntimeError("Tidak ada split test di dataset.")
    tr = corpus[~corpus["split"].isin(tests)]
    pd.DataFrame({"reference_clean": tr["reference_answer"], "student_clean": tr["student_answer"],
                  "normalized_label": tr["normalized_label"]}).to_csv(PAIRS, index=False)
    say(f"      Split test: {tests}; training pairs (train saja): {len(tr)}")
    return tests


def langkah3_latih(tests):
    if os.path.exists(os.path.join(MODEL, "model.safetensors")):
        say(f"[3/5] Model sudah ada, pelatihan dilewati: {MODEL}")
        return
    say("[3/5] Melatih SBERT-EduAdapt dari split train saja (5 epoch, batch 16, lr 2e-5, seed 42).")
    say("      Di CPU biasanya 30-90 menit. Biarkan jendela terbuka; laptop jangan sampai sleep.")
    t = time.time()
    code = jalankan([sys.executable, "-u", os.path.join(HERE, "train_eduadapt_heldout.py"),
                     "--data", CORPUS, "--test-splits", ",".join(tests), "--out", MODEL])
    if code != 0:
        raise RuntimeError("Pelatihan gagal. Kirim log_pipeline.txt ke Claude.")
    say(f"      Pelatihan selesai dalam {(time.time() - t) / 60:.1f} menit.")


def langkah4_ablation(tests):
    say("[4/5] Menjalankan ablation...")
    runs = [("all", ",".join(tests))] + [(s, s) for s in tests]
    for name, ts in runs:
        out = os.path.join(OUT, f"results_{name}")
        if os.path.exists(os.path.join(out, "run_metadata.json")):
            say(f"      results_{name} sudah ada, dilewati.")
            continue
        say(f"\n      ===== Test split: {ts} =====")
        code = jalankan([sys.executable, "-u", os.path.join(HERE, "run_ablation.py"),
                         "--data", CORPUS, "--eduadapt", MODEL, "--training-pairs", PAIRS,
                         "--test-splits", ts, "--allow-leakage", "--out", out])
        if code != 0:
            raise RuntimeError(f"Ablation untuk {ts} gagal. Kirim log_pipeline.txt ke Claude.")


def langkah5_kemas():
    shutil.copy(CORPUS, OUT)
    shutil.copy(LOG, OUT)
    tmp = os.path.join(HERE, "_kemas")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(OUT, tmp, ignore=shutil.ignore_patterns("cache"))
    zip_path = shutil.make_archive(os.path.join(HERE, "sear_results"), "zip", tmp)
    shutil.rmtree(tmp)
    say(f"[5/5] SELESAI. Kirim file ini ke Claude:\n      {zip_path}")


def main():
    say("=" * 70)
    say(f" SEAR Paper 1 - pipeline SciEntsBank   ({time.strftime('%Y-%m-%d %H:%M')})")
    say("=" * 70)
    say(f"Python {sys.version.split()[0]} di {sys.executable}")
    if sys.version_info < (3, 9):
        raise RuntimeError("Butuh Python 3.9 atau lebih baru.")
    langkah1_paket()
    os.makedirs(OUT, exist_ok=True)
    tests = langkah2_korpus()
    langkah3_latih(tests)
    langkah4_ablation(tests)
    langkah5_kemas()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        say("\n" + "!" * 70 + "\nTERJADI MASALAH:\n" + str(e) + "\n" + "-" * 70)
        say(traceback.format_exc())
        say(f"Kirim file ini ke Claude: {LOG}")
    finally:
        _log.close()
        try:
            input("\nTekan Enter untuk menutup jendela ini...")
        except EOFError:
            pass
