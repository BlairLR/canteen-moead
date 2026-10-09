# -*- coding: utf-8 -*-
"""一次性研究脚本：多服务器尝试获取南湖校区 OSM 数据（WGS-84）。

只做网络获取与结果打印/存档，不修改任何模型文件。
"""

import json
import os

import urllib.request
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "georef")
os.makedirs(OUT, exist_ok=True)

QL = """[out:json][timeout:40];
(
  nwr["barrier"="gate"](41.755,123.405,41.778,123.435);
  nwr["barrier"="wall"](41.755,123.405,41.778,123.435);
  nwr["name"~"东北大学"](41.755,123.405,41.778,123.435);
  nwr["amenity"="university"](41.750,123.400,41.785,123.445);
);
out center tags;"""

SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

data = urllib.parse.urlencode({"data": QL}).encode()
for server in SERVERS:
    try:
        req = urllib.request.Request(server, data=data, headers={"User-Agent": "neu-canteen-research/1.0 (student project)"})
        with urllib.request.urlopen(req, timeout=75) as r:
            j = json.load(r)
        els = j.get("elements", [])
        print(f"[OK] {server}  elements={len(els)}")
        with open(os.path.join(OUT, "osm_raw.json"), "w", encoding="utf-8") as fh:
            json.dump(j, fh, ensure_ascii=False, indent=1)
        for e in els[:80]:
            lat = e.get("lat") or (e.get("center") or {}).get("lat")
            lon = e.get("lon") or (e.get("center") or {}).get("lon")
            tags = e.get("tags", {})
            name = tags.get("name", "")
            keys = {k: v for k, v in tags.items() if k in ("barrier", "amenity", "building", "gate", "operator", "access")}
            print(f"  {e['type']}/{e['id']} ({lat}, {lon}) name={name!r} {keys}")
        break
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {server}: {type(exc).__name__}: {exc}")
else:
    print("全部 Overpass 服务器失败")
