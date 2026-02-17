# Wan2GP Timeline Editor Plugin

Plugin Wan2GP orienté **Timeline Editor multi-pistes** (style NLE / Premiere), structuré pour être **installable via URL GitHub** dans le manager de plugins Wan2GP.

## Exigences Wan2GP pour un repo “installable”

Ce dépôt respecte les prérequis clés côté loader Wan2GP :

1. `plugin.py` à la racine du repo.
2. dossier importable en package Python (`__init__.py` présent).
3. métadonnées plugin dans `plugin_info.json`.
4. dépendances optionnelles via `requirements.txt` (non requis ici pour le scaffold).

## Structure actuelle (MVP installable)

```text
Wan2GP-Timeline-Editor-Plugin/
  __init__.py
  plugin.py
  plugin_info.json
  README.md
  CHANGELOG.md
  LICENSE
```

## Installation (Wan2GP)

1. Ouvrir l’onglet **Plugins** dans Wan2GP.
2. Installer via l’URL GitHub de ce repository.
3. Redémarrer Wan2GP si demandé.
4. Vérifier la présence de l’onglet **Timeline Editor**.

## Roadmap courte

- **Phase 1 (MVP)** : tab UI + base plugin (ce repo).
- **Phase 2** : media bin + timeline clips + trim/cut.
- **Phase 3** : transitions (`xfade`, `acrossfade`) et compositing multi-tracks.
- **Phase 4** : export presets (H.264/H.265, vertical 9:16) + proxy preview.

## Références

- Wan2GP main repo: https://github.com/deepbeepmeep/Wan2GP
- Exemples de plugins Wan2GP (gallery, etc.)
