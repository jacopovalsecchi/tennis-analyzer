# Prova MediaPipe

Primo passo del progetto: vedere se MediaPipe tiene spalle, anche, gomiti e polsi su un video filmato da dietro. Lo script non calcola angoli e non dice se il colpo è giusto.

## Cosa fai tu

1. Prendi un video di 10–15 secondi, da dietro il giocatore, con un dritto e un rovescio. Il giocatore deve essere grande nell’inquadratura. Per questa prova meglio un solo corpo in quadro.
2. Copialo in `videos/`. Il nome può essere qualsiasi, per esempio `videos/sample.mp4`.
3. Nel terminale, da questa cartella:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python try_mediapipe.py videos/sample.mp4
```

La prima esecuzione scarica il modello `pose_landmarker_full` in `models/` (circa 9 MB). Poi legge il video e scrive due file in `output/`:

- `sample_overlay.mp4`, il video con lo scheletro sopra
- `sample_keypoints.csv`, le coordinate frame per frame

4. Apri l’overlay e guardalo sul dritto e sul rovescio, soprattutto nel frame più veloce.

Verde significa visibilità almeno 0,5. Rosso significa punto incerto. In alto a sinistra ci sono le visibilità dei due polsi. Sinistra e destra sono quelle del giocatore: con la camera da dietro, il polso destro sta sul lato destro dell’immagine.

## Quando il passo 1 è superato

Spalle, anche, gomito e polso del braccio che colpisce restano sul corpo per tutto il gesto, e il polso non salta al contatto. Il riepilogo nel terminale deve avere una copertura alta e pochi frame di polso sotto 0,5 durante il colpo.

Se il polso schizza via o lo scheletro si attacca a un’altra persona, il passo 1 per MediaPipe non è chiuso. Si ripete lo stesso video con un modello più pesante, e se ancora fallisce si passa a YOLO pose. Fermati e guarda il video prima di andare avanti.

## Passo 2 — feature dei colpi

Con il virtualenv già attivo:

```bash
python extract_features.py output/sample_keypoints.csv --video videos/sample.mov
```

Lo script cerca i picchi di velocità del polso e, per ogni colpo, calcola angolo del gomito, angolo fra avambraccio e busto, velocità, distanze e direzione dell’oscillazione. La punta della racchetta è il prolungamento dell’avambraccio oltre il polso: MediaPipe non vede la racchetta.

Scrive tre file in `output/`:

- `sample_strokes.csv`, una riga per colpo
- `sample_stroke_frames.csv`, gli stessi angoli frame per frame dentro il colpo
- `sample_features.mp4`, il video con la linea gialla dell’avambraccio e della racchetta stimata

Apri quel video. La riga in tabella vale se, in quel momento, la linea gialla sta sull’avambraccio e la punta prosegue oltre la mano. Se compare la scritta «polso incerto», quel frame è stato scartato perché il polso ha saltato. Le distanze sono in lunghezze del busto, la velocità in busti al secondo.
