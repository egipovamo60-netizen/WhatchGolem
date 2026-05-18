"""
Патчит quantization_config в config.json модели Qwen3.5-4B-AWQ:
заменяет 're:.*self_attn' на точные target-слои q_proj/k_proj/v_proj/o_proj.
Бэкапит оригинальный файл как config.json.bak
"""
import json
import os
import shutil

config_path = os.path.join(os.path.dirname(__file__), "Models", "Qwen3.5-4B-AWQ", "config.json")
backup_path = config_path + ".bak"

if not os.path.exists(backup_path):
    shutil.copy2(config_path, backup_path)
    print(f"Бэкап создан: {backup_path}")
else:
    print(f"Бэкап уже существует: {backup_path}")

with open(config_path, encoding="utf-8") as f:
    cfg = json.load(f)

qcfg = cfg.get("quantization_config", {})
groups = qcfg.get("config_groups", {})

if "group_1" in groups:
    old_targets = groups["group_1"].get("targets", [])
    new_targets = []
    for t in old_targets:
        if t == "re:.*self_attn":
            # раскрываем в точные подслои
            new_targets.extend([
                "re:.*self_attn\\.q_proj",
                "re:.*self_attn\\.k_proj",
                "re:.*self_attn\\.v_proj",
                "re:.*self_attn\\.o_proj",
            ])
        else:
            new_targets.append(t)
    groups["group_1"]["targets"] = new_targets
    print(f"group_1 targets обновлены: {old_targets} -> {new_targets}")
else:
    print("group_1 не найдена, патч не нужен")

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
print("config.json обновлён")
