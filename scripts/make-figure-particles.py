# Particle portrait for the hero: an authored profile bust (figure-profile.svg, rendered at 3x to profile3x.png)
# sampled as clean offset rings, each smoothed more than the last. Output: int16 (x, y) pairs, height-normalised, y up.
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage import measure
im = np.asarray(Image.open('profile3x.png').convert('L'), dtype=np.float32)
H, W = im.shape
inside = im < 128
sd0 = distance_transform_edt(inside) - distance_transform_edt(~inside)
rng = np.random.default_rng(5)
S = 44.0
levels = [1.5] + [S * k for k in range(1, 6)]
pts = []
for li, lv in enumerate(levels):
    sd = gaussian_filter(sd0, 2.0 + li * 7.0)
    step = 5.0 if li == 0 else 6.2
    for c in measure.find_contours(sd, lv):
        seg = np.diff(c, axis=0); L = np.hypot(*seg.T); tot = L.sum()
        if tot < 80: continue
        cum = np.concatenate([[0], np.cumsum(L)])
        s = np.arange(rng.random() * step, tot, step)
        y = np.interp(s, cum, c[:, 0]); x = np.interp(s, cum, c[:, 1])
        j = rng.normal(0, 0.55, len(s))
        dy = np.gradient(y); dx = np.gradient(x); nrm = np.hypot(dx, dy) + 1e-6
        x = x + j * (-dy / nrm); y = y + j * (dx / nrm)
        keep = rng.random(len(s)) < np.clip((0.985 - y / H) / 0.2, 0, 1)
        pts.extend(zip(x[keep], y[keep]))
pts = np.array(pts)
nx = (pts[:, 0] - W / 2) / H; ny = (H / 2 - pts[:, 1]) / H
open('silhouette.bin', 'wb').write(np.stack([np.round(nx * 20000), np.round(ny * 20000)], 1).astype('<i2').tobytes())
prev = Image.new('L', (W // 3, H // 3), 247); dr = ImageDraw.Draw(prev)
for x, y in pts / 3: dr.ellipse((x - 0.7, y - 0.7, x + 0.7, y + 0.7), fill=35)
prev.save('preview.png')
print('points', len(pts))
