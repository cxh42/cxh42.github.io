# Particle portrait for the hero, sampled from the owner's headshot. The photo is never committed or shipped:
# pass its path. The subject is cut from the plain studio backdrop, its outline lightly smoothed, and points are
# drawn with density set by heavily blurred tone (no facial features survive), a denser rim and a fading base.
# Output: public/data/silhouette.bin, int16 (x, y) pairs, height-normalised, y up; plus a preview PNG.
# Run: uv run --no-project --with numpy --with scipy --with scikit-image --with pillow \
#        python scripts/make-portrait-particles.py <photo> [preview.png]
import sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.color import rgb2lab

N = 11000
TOP, BOTTOM, HALF_W = 0.449, -0.497, 0.375  # the frame portrait.ts lays out
CUT = 0.87  # keep this much of the figure's height, so the head reads at the same size as before
im = np.asarray(Image.open(sys.argv[1]).convert('RGB')) / 255.0
H, W = im.shape[:2]
lab = rgb2lab(im); L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

# Backdrop: the mid-tone blue-grey region connected to the top and sides.
bgish = ndi.binary_opening((L > 40) & (L < 80) & (b < -8) & (np.abs(a) < 6), iterations=2)
lab_, _ = ndi.label(bgish)
edge = np.unique(np.concatenate([lab_[0], lab_[:, 0], lab_[:, -1]]))
fg = ndi.binary_opening(ndi.binary_fill_holes(~np.isin(lab_, edge[edge > 0])), iterations=3)
lab_, n = ndi.label(fg)
fg = lab_ == 1 + np.argmax(ndi.sum(fg, lab_, range(1, n + 1)))
sd = ndi.distance_transform_edt(fg) - ndi.distance_transform_edt(~fg)
fg = ndi.gaussian_filter(sd, 3.0) > 0

# Tone inside the figure only (normalised convolution), blurred far past any facial detail.
sig = 0.035 * H
w = ndi.gaussian_filter(fg.astype(float), sig)
tone = ndi.gaussian_filter(np.where(fg, L, 0.0), sig) / np.maximum(w, 1e-6)
dark = np.clip(1 - tone / 100, 0, 1)
rim = np.exp(-ndi.distance_transform_edt(fg) / (0.007 * H))
ys, xs = np.nonzero(fg)
top = ys.min()
bot = top + CUT * (H - top)
scale = (bot - top) / (TOP - BOTTOM)
head = fg[top + int(0.3 * (bot - top))]
cx = np.nonzero(head)[0].mean()  # the head's axis, not the frame's
yy, xx = np.mgrid[0:H, 0:W]
v = (yy - top) / (bot - top)
base = np.clip((0.985 - v) / 0.2, 0, 1) * np.interp(v, [0, 0.5, 1], [1, 1, 0.6])  # the jacket must not outweigh the head
dens = (0.08 + 0.92 * dark ** 2 + 0.55 * rim) * base * fg * (np.abs(xx - cx) / scale <= HALF_W)

rng = np.random.default_rng(7)
p = dens.ravel() / dens.sum()
idx = rng.choice(p.size, N, replace=False, p=p)
py, px = np.divmod(idx, W)
px = px + rng.random(N); py = py + rng.random(N)
nx = (px - cx) / scale; ny = TOP - (py - top) / scale
open('public/data/silhouette.bin', 'wb').write(
    np.stack([np.round(nx * 20000), np.round(ny * 20000)], 1).astype('<i2').tobytes())

if len(sys.argv) > 2:
    S = 1000
    prev = Image.new('L', (S, S), 247); dr = ImageDraw.Draw(prev)
    for x, y in zip(S / 2 + nx * S * 0.95, S / 2 - ny * S * 0.95):
        dr.ellipse((x - 1, y - 1, x + 1, y + 1), fill=40)
    prev.save(sys.argv[2])
print('points', N, 'x', nx.min().round(3), nx.max().round(3), 'y', ny.min().round(3), ny.max().round(3))
