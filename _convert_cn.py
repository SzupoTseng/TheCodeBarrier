#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""繁體中文稿 -> 简体中文稿（tw2sp：含台灣用語轉大陸用語）。輸出到 cn/。
   同時把 _build.py 轉成 cn/_build.py（簡體標題/檔名），並複製封面。
   需在有 opencc 的環境執行（Windows python）。"""
import os, re, glob, sys, shutil, opencc
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

c = opencc.OpenCC('tw2sp')
EXTRA = {'牠': '它', '妳': '你', '祂': '它'}
def post(t):
    for k, v in EXTRA.items():
        t = t.replace(k, v)
    return t

BOOK = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BOOK, 'cn')
os.makedirs(OUT, exist_ok=True)

def src_files():
    fs = []
    pre = glob.glob(os.path.join(BOOK, '00_*.md'))
    if pre:
        fs.append(pre[0])
    for n in range(1, 50):
        cand = glob.glob(os.path.join(BOOK, '%02d_*.md' % n))
        if cand:
            fs.append(cand[0])
    for apx in sorted(glob.glob(os.path.join(BOOK, '附錄*_*.md'))):
        fs.append(apx)
    return fs

n = 0
for f in src_files():
    base = os.path.basename(f)
    text = open(f, encoding='utf-8').read()
    cn_text = post(c.convert(text))
    cn_base = post(c.convert(base))
    with open(os.path.join(OUT, cn_base), 'w', encoding='utf-8') as w:
        w.write(cn_text)
    print(base, '->', cn_base)
    n += 1

# 轉換 _build.py -> cn/_build.py（簡體標題、檔名 glob、輸出名一併簡體化）
bsrc = open(os.path.join(BOOK, '_build.py'), encoding='utf-8').read()
bcn = post(c.convert(bsrc))
with open(os.path.join(OUT, '_build.py'), 'w', encoding='utf-8') as w:
    w.write(bcn)
print('_build.py -> cn/_build.py')

# 複製封面到 cn/（讓 cn 構建與根目錄一致地找到封面）
for cov in ('TheCodeBarrier.png', 'CodeBarrier.png'):
    src = os.path.join(BOOK, cov)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, cov))
        print('cover ->', os.path.join('cn', cov))
        break

print('converted %d md files + build + cover into %s' % (n, OUT))
