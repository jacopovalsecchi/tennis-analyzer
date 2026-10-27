# Piano decisionale — analizzatore di colpi

Ordine di lavoro: una decisione alla volta. Non si costruisce il sito, i container o il classificatore finché i passi precedenti non sono chiusi.

## Roadmap

**1. Scegliere la libreria di pose e provarla su un video.**
Si prende un video breve, filmato da dietro il giocatore, e si fa girare una libreria sola. Si guarda se spalle, anche, gomiti e polsi restano agganciati durante un dritto e un rovescio. Se i punti saltano o si perdono sul movimento veloce, quella libreria non va bene. Si cambia libreria e si riprova. Nient’altro parte prima di questo sì.

**2. Estrarre le feature.**
Dai punti nel tempo si calcolano angoli, velocità e distanze tra spalla, gomito, polso e posizione stimata della racchetta. La racchetta non è un keypoint del corpo: si stima a partire dal polso, per esempio prolungando l’avambraccio. Si salva una tabella per colpo e si controlla a occhio che i numeri si muovano come il gesto.

**3. Decidere come giudicare il colpo.**
Due strade, una sola da tenere per la prima versione.

- Regole scritte: angoli, velocità e distanze confrontati con intervalli tipici di un bel colpo.
- Classificatore supervisionato: stesso vettore di feature, addestrato su video etichettati da un maestro.

Le regole bastano per partire e si spiegano. Il classificatore ha senso quando le regole sbagliano troppo e ci sono abbastanza clip etichettate.

**4. Solo allora, la pipeline intera.**
Upload, coda, worker GPU, API e pagina delle statistiche. A quel punto pose, feature e criterio di giudizio sono già scelti. Il resto trasporta quei risultati, non li inventa.

## Le tre librerie di pose

Tutte e tre sono open source e gratuite. Tutte restituiscono coordinate di articolazioni, frame dopo frame. Nessuna conosce dritto, rovescio o “colpo corretto”.

### MediaPipe

Sito: [mediapipe.dev](https://mediapipe.dev)

- Progetto aperto di Google, uso libero.
- 33 keypoint: spalle, anche, gomiti, polsi e il resto del corpo, più del necessario per il braccio.
- Leggera. Gira su CPU, utile per la prova sul video campione senza scheda video.
- Per la racchetta c’è solo il polso. La posizione della racchetta va dedotta, di solito prolungando la linea gomito–polso.
- Sul gesto molto veloce i punti sono meno stabili: il polso può saltare proprio nel momento del contatto.

### YOLO pose (Ultralytics)

Sito: [ultralytics.com](https://ultralytics.com)

- Progetto aperto, uso libero. Il modello di pose è quello della famiglia YOLO distribuito da Ultralytics.
- 17 keypoint. Coprono spalle, anche, gomiti e polsi. Meno dettagli di MediaPipe, abbastanza per braccio e busto.
- Più stabile sui corpi che si muovono in fretta, che è il caso di un’oscillazione.
- In pratica chiede una GPU. Senza scheda video la prova sul video campione diventa lenta.
- Anche qui non esiste un punto “racchetta”: si stima come con le altre librerie.

### OpenPose

Repository: [github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

- Progetto aperto della Carnegie Mellon, uso libero.
- Pensato per più persone nello stesso frame. Utile se in quadro entrano avversario o compagni, superfluo se il video è un solo giocatore di spalle.
- Più pesante e più lento delle altre due, anche con GPU.
- Installazione più lunga: dipendenze, build, modelli. Per il primo video campione è la strada con più attrito.
- Stesso limite sulla racchetta: articolazioni del corpo, poi una stima indiretta.

Scelta pratica per il passo 1: provare MediaPipe sul video campione, perché parte su CPU. Se il polso si perde sul colpo veloce, ripetere lo stesso video con YOLO pose. OpenPose resta la terza prova, se le prime due non tengono la persona giusta.

## Cosa fa la pose, e cosa no

La pose estimation scrive solo coordinate nel tempo: dove sono spalla, gomito, polso, anca, a ogni frame. Non dice se il colpo è giusto.

Quel giudizio sta sopra, in un secondo strato.

Con le regole, si confrontano angoli, velocità e distanze con fasce tipiche di un buon colpo. Esempio: l’angolo tra racchetta stimata e busto al contatto, la direzione del polso, quanto il gomito è aperto. Fuori fascia, il colpo viene segnato. Le fasce si tarano con un maestro e si leggono senza un dataset.

Con il classificatore supervisionato, le stesse misure diventano l’input di un modello addestrato su video già etichettati dal maestro, per esempio dritto corretto, rovescio corretto, gesto da correggere. Serve una raccolta di clip con etichetta. Senza quelle etichette il modello non ha niente da imparare.

Il passo 3 è la scelta tra questi due strati. La libreria di pose, sotto, resta la stessa.
