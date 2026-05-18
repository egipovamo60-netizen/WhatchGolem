"""
Патч 2: убирает group_0 (linear_attn 8-bit int8) из quantization_config
т.к. compressed-tensors не может распаковать веса с group_size=32 при dim=4.
"""
import json
import os

config_path = os.path.join(os.path.dirname(__file__), "Models", "Qwen3.5-4B-AWQ", "config.json")

with open(config_path, encoding="utf-8") as f:
    cfg = json.load(f)

qcfg = cfg.get("quantization_config", {})
groups = qcfg.get("config_groups", {})

if "group_0" in groups:
    old = groups.pop("group_0")
    print("Удалена group_0 (linear_attn int8):", old.get("targets"))
else:
    print("group_0 не найдена")

print("Оставшиеся группы:", list(groups.keys()))
for gname, gdata in groups.items():
    print(f"  {gname}: targets={gdata.get('targets')}, bits={gdata.get('weights', {}).get('num_bits')}")

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
print("config.json обновлён (group_0 удалена)")
