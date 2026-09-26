"""Builds src/data/regions.json for the visitor globe: {country: {region key: [lat, lon]}}.

GoatCounter's API reports a visit's region by its English name only (US "California", CN "Henan"), so each
Natural Earth admin-1 unit is indexed under the normalised forms of its names; ISO 3166-2 codes (US-TX) are
indexed too. Where Natural Earth is finer than a GeoIP region (French departments, English counties), the
parent region is also rolled up, weighted by area. Keys are made by norm(), mirrored in Visitors.astro.

Source: Natural Earth 1:10m admin-1 states and provinces (public domain).
  curl -LO https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_1_states_provinces.geojson
  python3 scripts/make-region-centroids.py ne_10m_admin_1_states_provinces.geojson
"""
import json
import math
import re
import unicodedata
import sys
from collections import defaultdict

src = sys.argv[1]
feats = json.load(open(src))['features']

acc = defaultdict(lambda: [0.0, 0.0, 0.0])  # (country, key) -> [sum w*lat, sum w*lon, sum w]
parents = defaultdict(lambda: [0.0, 0.0, 0.0])  # rolled-up parent regions, used only where no unit has the key


DROP = {'province', 'prefecture', 'region', 'state', 'municipality', 'oblast', 'governorate', 'autonomous',
        'special', 'administrative', 'of', 'the', 'city', 'metropolitan', 'district', 'territory', 'capital'}


def norm(name):
    s = unicodedata.normalize('NFKD', name)
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower().replace('&', ' and ')
    words = [w for w in re.split(r'[^a-z0-9]+', s) if w]
    kept = [w for w in words if w not in DROP]
    return ''.join(kept or words)


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


def add(key, lat, lon, w, into=acc):
    a = into[key]
    a[0] += w * lat
    a[1] += w * lon
    a[2] += w


for f in feats:
    q = f['properties']
    cc, lat, lon = q['iso_a2'], q['latitude'], q['longitude']
    if not cc or len(cc) != 2 or lat is None or lon is None:
        continue
    w = area(f['geometry']) + 1e-6 if f.get('geometry') else 1e-6
    keys = set()
    iso = (q.get('iso_3166_2') or '').strip()
    if iso.startswith(cc + '-') and not iso.endswith('~'):
        keys.add(iso)
    names = [q.get(f) or '' for f in ('name', 'name_en', 'gn_name', 'woe_name')]
    names += (q.get('name_alt') or '').split('|')
    keys |= {norm(v) for v in names if v.strip()}
    for k in keys - {''}:
        add((cc, k), lat, lon, w)
    # Parent regions, by name and in ISO or HASC form.
    if q.get('region'):
        add((cc, norm(q['region'])), lat, lon, w, parents)
    rc = (q.get('region_cod') or '').strip()
    for sep in ('-', '.'):
        if rc.startswith(cc + sep) and len(rc) > 3:
            add((cc, cc + '-' + rc[3:]), lat, lon, w, parents)
            break

for k, a in parents.items():
    if k not in acc:
        acc[k] = a

# The four nations of the United Kingdom, which Natural Earth splits into counties and unitary authorities.
for code, name, lat, lon in [('GB-ENG', 'England', 52.6, -1.5), ('GB-SCT', 'Scotland', 56.8, -4.2),
                             ('GB-WLS', 'Wales', 52.3, -3.7), ('GB-NIR', 'Northern Ireland', 54.6, -6.7)]:
    acc[('GB', code)] = acc[('GB', norm(name))] = [lat, lon, 1]

out = defaultdict(dict)
for (cc, k), a in sorted(acc.items()):
    if k:
        out[cc][k] = [round(a[0] / a[2], 1), round(a[1] / a[2], 1)]
json.dump(out, open('src/data/regions.json', 'w'), separators=(',', ':'), ensure_ascii=False)
print(f'{sum(len(v) for v in out.values())} keys in {len(out)} countries')
