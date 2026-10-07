import os
from collections import Counter

ROOT = "."          # run from your project root
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

print("=== PROJECT FILES (top 2 levels, excluding image files) ===")
for dp, dn, fn in os.walk(ROOT):
    depth = dp[len(ROOT):].count(os.sep)
    if depth > 1 or any(s in dp for s in (".git", "__pycache__", "venv", ".venv", "node_modules")):
        continue
    for f in fn:
        if os.path.splitext(f)[1].lower() not in IMG_EXT:
            print(os.path.join(dp, f))

print("\n=== CANDIDATE DATA FOLDERS (folders whose subfolders contain images) ===")
found = []
for dp, dn, fn in os.walk(ROOT):
    if any(s in dp for s in (".git", "venv", ".venv", "node_modules")):
        continue
    subs = [d for d in dn if any(os.path.splitext(f)[1].lower() in IMG_EXT
                                 for f in os.listdir(os.path.join(dp, d)))]
    if len(subs) >= 2:
        found.append((dp, sorted(subs)))

for dp, subs in found:
    print(f"\nFOLDER: {dp}")
    print(f"  Number of classes: {len(subs)}")
    total = 0
    exts = Counter()
    sizes = Counter()
    for s in subs:
        p = os.path.join(dp, s)
        files = [f for f in os.listdir(p) if os.path.splitext(f)[1].lower() in IMG_EXT]
        total += len(files)
        for f in files:
            exts[os.path.splitext(f)[1].lower()] += 1
        print(f"  {s}: {len(files)} images")
    print(f"  Total images: {total}")
    print(f"  Image formats: {dict(exts)}")
    try:
        from PIL import Image
        for s in subs:
            p = os.path.join(dp, s)
            for f in os.listdir(p)[:20]:
                if os.path.splitext(f)[1].lower() in IMG_EXT:
                    with Image.open(os.path.join(p, f)) as im:
                        sizes[(im.size, im.mode)] += 1
        print(f"  Sample sizes/modes (first 20 per class): {dict(sizes)}")
    except ImportError:
        print("  (Pillow not installed - skipping size check)")

print("\n=== NOTE ===")
print("Send me this entire output. If the Training and Validation folders were not both found, tell me their exact paths.")