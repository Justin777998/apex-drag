# cargen – Autos für Nachtmeile

Dieser Zweig erzeugt Automodelle automatisch über GitHub Actions:

1. **Konzeptbild:** FLUX.1 [schnell] (Black Forest Labs, Lizenz Apache-2.0)
2. **3D-Modell:** TRELLIS (Microsoft, Lizenz MIT), ersatzweise TripoSR (MIT)

Die Beschreibungen stehen in `prompts.json`. Ein Push auf diesen Zweig startet den Lauf, die Ergebnisse landen in `out/<id>/`
(`concept.png`, `model.glb`, `info.json`). Ein Protokoll liegt in `out/log.txt`.

Es werden keine echten Marken, Logos oder Modellnamen verwendet.
