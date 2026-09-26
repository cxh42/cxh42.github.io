"""Builds src/data/regions.json: ISO 3166-2 subdivision code -> [lat, lon], for the visitor globe.

GoatCounter reports a visit's region as the first-level subdivision its GeoIP database gives (US-TX, CN-BJ,
FR-IDF, IT-25, GB-ENG...). Natural Earth's admin-1 layer is finer in some countries (French departments,
Italian provinces, English counties), so those are also rolled up into their parent region, weighted by area.

Source: Natural Earth 1:10m admin-1 states and provinces (public domain).
  curl -LO https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_1_states_provinces.geojson
  python3 scripts/make-region-centroids.py ne_10m_admin_1_states_provinces.geojson
"""
import json
import math
import sys
from collections import defaultdict

src = sys.argv[1]
feats = json.load(open(src))['features']

acc = defaultdict(lambda: [0.0, 0.0, 0.0])  # code -> [sum w*lat, sum w*lon, sum w]


def area(geom):
    # Rough planar area of the outer rings, in square degrees scaled by cos(latitude): enough to weigh parts.
    polys = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
    total = 0.0
    for poly in polys:
        ring = poly[0]
        s = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]))
        mean_lat = sum(y for _, y in ring) / len(ring)
        total += abs(s) / 2 * math.cos(math.radians(mean_lat))
    return total


def add(code, lat, lon, w):
    a = acc[code]
    a[0] += w * lat
    a[1] += w * lon
    a[2] += w


for f in feats:
    q = f['properties']
    cc, lat, lon = q['iso_a2'], q['latitude'], q['longitude']
    if not cc or len(cc) != 2 or lat is None or lon is None:
        continue
    w = area(f['geometry']) + 1e-6 if f.get('geometry') else 1e-6
    iso = (q.get('iso_3166_2') or '').strip()
    if iso.startswith(cc + '-') and not iso.endswith('~'):
        add(iso, lat, lon, w)
    # Parent regions, where Natural Earth records them in ISO or HASC form.
    rc = (q.get('region_cod') or '').strip()
    for sep in ('-', '.'):
        if rc.startswith(cc + sep) and len(rc) > 3:
            add(cc + '-' + rc[3:], lat, lon, w)
            break

# The four nations of the United Kingdom, which Natural Earth splits into counties and unitary authorities.
for code, lat, lon in [('GB-ENG', 52.6, -1.5), ('GB-SCT', 56.8, -4.2), ('GB-WLS', 52.3, -3.7), ('GB-NIR', 54.6, -6.7)]:
    acc[code] = [lat, lon, 1]

out = {k: [round(a[0] / a[2], 1), round(a[1] / a[2], 1)] for k, a in sorted(acc.items())}
json.dump(out, open('src/data/regions.json', 'w'), separators=(',', ':'))
print(f'{len(out)} regions')
