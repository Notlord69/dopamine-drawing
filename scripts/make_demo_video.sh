#!/usr/bin/env bash
# Cut the demo video and GIF from the videos the week 9 / week 10 notebooks recorded. Nothing is
# re-simulated: every clip is a frame range of an existing video, with a caption strip added below.
#
#   scripts/make_demo_video.sh [DATA_DIR] [OUT_DIR]
#     DATA_DIR  holds week9/week9_pick_place_kaggle.mp4 and week10/week10_pick_place_kaggle.mp4
#               (default $FLYARM_DATA_DIR or <repo>/data)
#     OUT_DIR   default <repo>/docs/media  ->  demo.mp4, demo.gif
#
# Needs ffmpeg and Python with Pillow. The source videos are 1280x480 at 12.5 fps (half speed); each
# episode is (steps + 1) frames plus its last frame held for 10 frames (sim/video7.py). Frame ranges:
#   week10 video: connectome C4 seed 93000 (success)       frames   0-89
#                 connectome C4 seed 93011 (put down)      frames  90-249
#                 random network C4 seed 93000 (success)   frames 250-345
#                 connectome C6 seed 93000 (failure)       frames 346-505   (346-665 holds it twice)
#   week9 video:  connectome C4 seed 92004 (the press)     frames 185-344
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${1:-${FLYARM_DATA_DIR:-$REPO_ROOT/data}}"
OUT="${2:-$REPO_ROOT/docs/media}"
PY="${PYTHON:-python}"
W10="$DATA/week10/week10_pick_place_kaggle.mp4"
W9="$DATA/week9/week9_pick_place_kaggle.mp4"
for f in "$W10" "$W9"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"
"$PY" "$REPO_ROOT/scripts/make_demo_cards.py" "$TMP" --width 1280 --height 480 --strip 64
ENC=(-c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -r 25 -an)

card() {   # card PNG, seconds, output
  ffmpeg -v error -y -loop 1 -framerate 25 -t "$2" -i "$1" -vf "fade=in:0:8,fade=out:st=$(awk "BEGIN{print $2 - 0.32}"):d=0.32" "${ENC[@]}" "$3"
}
clip() {   # source, first frame, last frame, caption PNG, output
  ffmpeg -v error -y -i "$1" -i "$4" -filter_complex \
    "[0:v]trim=start_frame=$2:end_frame=$(($3 + 1)),setpts=PTS-STARTPTS,fps=25[v];[v][1:v]vstack=inputs=2,format=yuv420p[o]" \
    -map "[o]" "${ENC[@]}" "$5"
}

card "$TMP/card_title.png"   3 "$TMP/s01.mp4"
card "$TMP/card_diagram.png" 6 "$TMP/s02.mp4"
clip "$W10"   0  89 "$TMP/cap_connectome.png" "$TMP/s03.mp4"
clip "$W10" 250 345 "$TMP/cap_random.png"     "$TMP/s04.mp4"
clip "$W9"  185 344 "$TMP/cap_press.png"      "$TMP/s05.mp4"
clip "$W10"  90 249 "$TMP/cap_putdown.png"    "$TMP/s06.mp4"
clip "$W10" 346 505 "$TMP/cap_c6.png"         "$TMP/s07.mp4"
card "$TMP/card_results.png" 8 "$TMP/s08.mp4"
card "$TMP/card_credits.png" 5 "$TMP/s09.mp4"

for i in 01 02 03 04 05 06 07 08 09; do echo "file '$TMP/s$i.mp4'"; done > "$TMP/list.txt"
ffmpeg -v error -y -f concat -safe 0 -i "$TMP/list.txt" -c copy -movflags +faststart "$OUT/demo.mp4"

# GIF for the top of the README: the connectome's successful episode, 640 px wide, 12 fps
ffmpeg -v error -y -i "$W10" -filter_complex \
  "[0:v]trim=start_frame=0:end_frame=90,setpts=PTS-STARTPTS,fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" \
  "$OUT/demo.gif"

for f in "$OUT/demo.mp4" "$OUT/demo.gif"; do
  printf '%s  %s  %ss\n' "$f" "$(du -h "$f" | cut -f1)" "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")"
done
