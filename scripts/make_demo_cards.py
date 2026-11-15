#!/usr/bin/env python
"""Title, diagram, results and credits cards and the caption strips for the demo video (Pillow).

    python scripts/make_demo_cards.py OUT_DIR [--width 1280 --height 480 --strip 64]

Every number on the cards is from results/week10/week10_results_kaggle.json and
results/week10b/week10b_results_kaggle.json (see README / WRITEUP).
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
FONT_DIRS = ['/usr/share/fonts/truetype/dejavu', '/usr/share/fonts/dejavu', '/Library/Fonts', 'C:/Windows/Fonts']
BG, INK, MUTED, ACCENT, GOOD, BAD = (17, 19, 24), (240, 240, 236), (168, 170, 176), (235, 104, 52), (110, 231, 160), (255, 179, 138)

# caption strips under each clip: (file stem, text, colour of the leading tag)
CAPTIONS = {
    'cap_connectome': ("CONNECTOME", "Frozen fly-CNS model + linear readout (C4), fresh test seed 93000: placed.  46/50 on seeds 93000–93049.", GOOD),
    'cap_random': ("RANDOM WIRING", "Degree-matched random network (same neurons, degrees, weights, E/I): 50/50.  Other shuffles: 49, 45, 50.", GOOD),
    'cap_press': ("FAILURE: THE PRESS", "Hovers above the target still holding the cube. Same connectome + C4 readout (week 9 test seed 92004).", BAD),
    'cap_putdown': ("FAILURE", "Connectome, seed 93011: hovers over the target holding the cube; sets it down as the 6 s limit runs out, never releases.", BAD),
    'cap_c6': ("ABLATION C6", "Up-weighting the press in the readout fit broke the carry: lifts every time, places 0/50.", BAD),
}


def font(size, bold=False):
    names = ['DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf', 'arialbd.ttf' if bold else 'arial.ttf']
    for d in FONT_DIRS:
        for n in names:
            p = Path(d) / n
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def centred(d, y, text, f, fill, W):
    w = d.textlength(text, font=f)
    d.text(((W - w) / 2, y), text, font=f, fill=fill)


def card_title(W, H):
    im = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(im)
    centred(d, H * 0.33, "A fly's brain drives a robot arm", font(54, True), INK, W)
    centred(d, H * 0.33 + 78, "A frozen spiking model of the whole male Drosophila CNS (176,422 neurons) controls a",
            font(22), MUTED, W)
    centred(d, H * 0.33 + 108, "simulated Franka Panda through pick-and-place, read out by one linear map.",
            font(22), MUTED, W)
    return im


def card_diagram(W, H):
    im = Image.new('RGB', (W, H), (255, 255, 255))
    dg = Image.open(REPO / 'docs' / 'architecture.png').convert('RGB')
    dg.thumbnail((W, H))
    im.paste(dg, ((W - dg.width) // 2, (H - dg.height) // 2))
    return im


def card_results(W, H):
    im = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(im)
    d.text((70, 34), "Results on 50 fresh test seeds (93000–93049)", font=font(30, True), fill=INK)
    rows = [("policy", "success", "placement (median)"),
            ("connectome + linear readout (C4)", "46/50", "14.3 mm"),
            ("random network, shuffle 0 / 1 / 2 / 3", "50, 49, 45, 50", "5.4 / 10.8 / 9.4 / 6.0 mm"),
            ("scripted expert (the teacher)", "50/50", "2.6 mm"),
            ("no network: linear readout on the 8 signals", "0/50", "–"),
            ("feedback cut (input frozen), all five networks", "0/10 each", "never lifts")]
    f, fb = font(21), font(21, True)
    xs, y = (70, 640, 860), 96
    for i, r in enumerate(rows):
        for x, t in zip(xs, r):
            d.text((x, y), t, font=fb if i == 0 or (i == 1 and x == xs[1]) else f, fill=MUTED if i == 0 else INK)
        y += 40 if i else 46
        if i == 0:
            d.line((70, y - 8, W - 70, y - 8), fill=(70, 72, 78), width=1)
    y += 18
    d.text((70, y), "Pre-registered test (paired McNemar): the connectome is better than 0 of 4 random networks", font=f, fill=INK)
    d.text((70, y + 32), "and worse than 0 of 4. Three of the four place the cube more precisely.", font=f, fill=INK)
    d.text((70, y + 82), "The wiring isn't what makes it work.", font=font(30, True), fill=ACCENT)
    return im


def card_credits(W, H):
    im = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(im)
    lines = [("Credits", font(34, True), INK, 0),
             ("Connectome: FlyEM male CNS v1.0, CC-BY 4.0. HHMI Janelia FlyEM with the University of Cambridge,", font(20), INK, 22),
             ("the MRC Laboratory of Molecular Biology and Google Research. Berg et al., bioRxiv 2025.", font(20), INK, 0),
             ("Neuron model: Shiu et al. 2024, Nature, \"A Drosophila computational brain model reveals sensorimotor processing\".", font(20), INK, 14),
             ("Robot: MuJoCo (Apache-2.0) and the MuJoCo Menagerie Franka Emika Panda model (Apache-2.0).", font(20), INK, 14),
             ("Simulation, readout and video by the author. No endorsement by the data or model authors is implied.", font(18), MUTED, 30)]
    y = 80
    for t, f, c, gap in lines:
        y += gap
        d.text((70, y), t, font=f, fill=c)
        y += f.size + 12
    return im


def caption_strip(W, h, tag, text, colour):
    im = Image.new('RGB', (W, h), BG); d = ImageDraw.Draw(im)
    ft, fx = font(19, True), font(19)
    x = 22
    d.text((x, (h - 22) / 2), tag, font=ft, fill=colour)
    x += d.textlength(tag, font=ft) + 16
    words, line, lines = text.split(), '', []
    for w in words:                           # wrap to the strip width
        t = (line + ' ' + w).strip()
        if d.textlength(t, font=fx) > W - x - 22:
            lines.append(line); line = w
        else:
            line = t
    lines.append(line)
    y = (h - 26 * len(lines)) / 2
    for l in lines:
        d.text((x, y), l, font=fx, fill=INK); y += 26
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', type=Path)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--strip', type=int, default=64)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    W, H = a.width, a.height + a.strip                    # cards have the clips' full size (frame + strip)
    for name, fn in (('card_title', card_title), ('card_diagram', card_diagram),
                     ('card_results', card_results), ('card_credits', card_credits)):
        fn(W, H).save(a.out / f'{name}.png')
    for name, (tag, text, c) in CAPTIONS.items():
        caption_strip(W, a.strip, tag, text, c).save(a.out / f'{name}.png')
    print('cards in', a.out)


if __name__ == '__main__':
    main()
