#!/usr/bin/env python3
"""把 src/chapterN.txt 依分隔線切塊，分類後寫成可編譯的檔案。"""
import os
import re
import sys

ROOT = "/home/awe/disk/cuda-code"
SEP = re.compile(r"^-{50,}$")


def classify(text):
    if "__global__" in text or "cuda_runtime.h" in text or "cudaMalloc" in text \
            or "<<<" in text or "__device__" in text or "cublas" in text.lower():
        return "cu"
    if re.search(r"^\s*(import |from \w+ import )", text, re.M) or re.search(r"^def \w+", text, re.M):
        return "py"
    if "#include" in text:
        return "cpp"
    return "sh"


def slug(text, kind):
    if kind in ("cu", "cpp"):
        m = re.findall(r"__global__ void (\w+)", text)
        if m:
            # 取出現次數最多的 kernel 名
            best = max(set(m), key=lambda k: (m.count(k), -m.index(k)))
            return camel_to_snake(best)
        m = re.search(r"\b(\w+)\s*\([^)]*\)\s*\{", text)
        if m and m.group(1) not in ("main", "if", "for", "while"):
            return camel_to_snake(m.group(1))
        return "program"
    if kind == "py":
        m = re.findall(r"^def (\w+)", text, re.M)
        for name in m:
            if name not in ("run_command", "main"):
                return name
        return m[0] if m else "script"
    # shell: 用第一個有意義的指令
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tok = re.sub(r"[^a-zA-Z0-9_-]", "_", line.split()[0])
        return "cmds_" + tok[:20]
    return "notes"


def camel_to_snake(name):
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower()


def main():
    summary = []
    for ch in range(3, 13):
        src = os.path.join(ROOT, "src", f"chapter{ch}.txt")
        with open(src, encoding="utf-8") as f:
            raw = f.read()
        blocks, cur = [], []
        for line in raw.splitlines():
            if SEP.match(line.strip()):
                blocks.append("\n".join(cur))
                cur = []
            else:
                cur.append(line)
        blocks.append("\n".join(cur))
        blocks = [b.strip("\n") for b in blocks if b.strip()]

        outdir = os.path.join(ROOT, f"ch{ch:02d}")
        os.makedirs(outdir, exist_ok=True)
        used = set()
        for i, body in enumerate(blocks, 1):
            kind = classify(body)
            name = slug(body, kind)
            base = f"{i:02d}_{name}"
            while base in used:
                base += "_b"
            used.add(base)
            standalone = bool(re.search(r"\bint main\s*\(", body)) if kind in ("cu", "cpp") else True
            if kind in ("cu", "cpp") and not standalone:
                base += "_snippet"
            path = os.path.join(outdir, f"{base}.{kind}")
            header = f"// chapter{ch} 片段 {i}（自 src/chapter{ch}.txt 抽出）\n"
            if kind == "py":
                header = f'"""chapter{ch} 片段 {i}（自 src/chapter{ch}.txt 抽出）"""\n'
            elif kind == "sh":
                header = f"#!/usr/bin/env bash\n# chapter{ch} 片段 {i}：指令清單（參考用，請勿直接整份執行）\n"
            if kind in ("cu", "cpp") and not standalone:
                header += "// 注意：原文只是片段，沒有 main()，無法單獨編譯。\n"
            if kind == "sh":
                body = "\n".join(
                    ("# " + l if l.strip() and not l.strip().startswith("#") else l)
                    for l in body.splitlines()
                )
            with open(path, "w", encoding="utf-8") as f:
                f.write(header + body + "\n")
            summary.append((ch, i, kind, standalone, os.path.relpath(path, ROOT)))

    for ch, i, kind, standalone, path in summary:
        flag = "" if standalone else "  [片段]"
        print(f"ch{ch:02d} #{i:02d} {kind:3s} {path}{flag}")
    print(f"\n共 {len(summary)} 個區塊")


if __name__ == "__main__":
    main()
