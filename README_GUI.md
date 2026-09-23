# Crack & Detachment Segmentation -- finestra unica con l'immagine incorporata

Questa consegna sostituisce le due finestre separate (OpenCV + pannello
Tkinter) di una versione precedente con **una sola finestra vera**: la
foto vive dentro un canvas incorporato nella stessa finestra del menu e
della barra strumenti, come un normale programma desktop. Non è un
riavvio da zero: la logica di segmentazione (`handle_keyboard`,
`mouse_callback`, `render_scene`, l'A*, SIFT/omografia, l'export
LabelMe...) è rimasta **esattamente quella di sempre**, invariata riga
per riga. È cambiato solo *dove* il fotogramma finale viene mostrato e
*da dove* arrivano mouse e tastiera.

## File inclusi

```
smart_segmentation.py        <- modificato (6 hook aggiunti, vedi sotto)
config.py                    <- modificato (cartella dati sorgente/app pacchettizzata)
compare_and_filter_cracks.py <- INVARIATO
crack_segmentation_gui.py    <- la finestra unica (riscritta in questa consegna)
requirements.txt             <- dipendenze, incluso Pillow (nuovo) e PyInstaller
test_gui_shell_parity.py     <- 71 test sugli hook/process_keypress/run()
test_gui_panel_wiring.py     <- 70 test su menu, mouse, tastiera, canvas scorrevole, striscia di stato
test_resolve_script_dir.py   <- 8 test sulla cartella dati (sorgente vs app pacchettizzata)
tests_support/fake_tkinter.py<- stub minimale usato solo dai test (niente display qui)
packaging/
  crack_segmentation_gui.spec       <- spec PyInstaller (funziona su entrambi gli OS)
  build_mac.sh                       <- genera CrackSegmentation.app + .dmg
  build_windows.bat                  <- genera .exe + Setup.exe (se Inno Setup è installato)
  build_linux.sh                     <- genera l'eseguibile + pacchetto .deb
  crack_segmentation_installer.iss   <- script Inno Setup per il Setup.exe
```

**149 test in totale**, tutti verdi.

## Redo dopo Annulla-di-un'eliminazione: la correzione precedente era a metà

Segnalazione: "adesso undo funziona redo no". Avevo corretto solo metà
del ciclo -- Annulla dopo un click-to-delete ora riporta indietro la
crepa/il distacco, ma non avevo registrato nulla per un successivo
Ripristina, quindi premendolo non succedeva niente.

Capito il motivo: il meccanismo di Ripristina in questo programma
significa sempre "rimetti questo elemento" (è così che funziona per il
caso normale: disegni una crepa, Annulla la rimuove, Ripristina la
rimette). Ma per un'eliminazione diretta via click, il ciclo corretto è
l'opposto -- Annulla deve *rimettere* la crepa eliminata, e Ripristina
subito dopo deve *rieliminarla di nuovo*, non rimetterla una seconda
volta. Serviva quindi un'etichetta di cronologia nuova e distinta (non
una di quelle già esistenti, che significano tutte "rimetti"), con un
comportamento opposto nel codice di Ripristina. Corretto per crepe e
distacchi, con un ciclo completo verificato: click-to-delete -> Annulla
(ripristina) -> Ripristina (elimina di nuovo) -> di nuovo annullabile
esattamente come l'eliminazione originale.

**149 test in totale** ora, tutti verdi.

## Undo/Redo sulle fratture: trovato un terzo bug, probabilmente il più rilevante

Continuando la ricerca (senza aspettare la sequenza esatta che stavi
usando) ho controllato **ogni singolo punto** del codice dove una crepa
o un distacco viene modificato o eliminato, per vedere se Annulla aveva
davvero un caso corrispondente per ciascuno. Trovato un terzo bug,
probabilmente il più probabile di tutti perché nasce dall'interazione
più comune in assoluto:

**Cliccare vicino all'estremità di una crepa la elimina immediatamente**
(funzione già esistente, non una novità di queste consegne) -- e questa
eliminazione veniva registrata con l'etichetta `crack_delete`. Il
controllo l'aveva già previsto per Ripristina (Redo la riconosceva
correttamente), ma **Annulla non aveva mai avuto un caso per questa
etichetta specifica** -- veniva scartata in silenzio, senza fare nulla.
Risultato pratico: se durante il lavoro normale capita di cliccare per
sbaglio vicino all'estremità di una crepa (facile, dato che il margine di
tolleranza è di 22 pixel), quella crepa sparisce e Annulla non la
riporta indietro -- esattamente il sintomo di "undo non funziona più".
Corretto in modo simmetrico anche per i distacchi (stesso identico
problema, stessa funzione di click-to-delete).

**146 test in totale** ora, tutti verdi. In totale, in questa indagine,
ho trovato e corretto tre bug distinti (T, V, e ora questo) più probabile
di tutti perché il click-to-delete è un'interazione quotidiana, non un
comando specialistico. Se il problema persiste ancora dopo questa
correzione, la sequenza esatta di passaggi resta comunque la cosa più
utile che puoi darmi per continuare a restringere il campo.

## Undo/Redo sulle fratture: due bug reali trovati (ma serve conferma)

Segnalazione: "undo e redo sulle fratture non funzionano più". Sono
partito rifacendo un test end-to-end che passa per il vero percorso
tastiera → Tkinter → programma (non solo la logica isolata) per il caso
più semplice -- disegna una crepa, Annulla, Ripristina -- e quel percorso
**risulta ancora corretto**. Questo significa che il problema più comune
(quello già corretto in una consegna precedente) non si è ripresentato.

Continuando a cercare ho però trovato **due bug reali e distinti**,
entrambi preesistenti nel codice originale, che rendono Annulla
inaffidabile in scenari più specifici -- possibile causa di quello che
hai visto, ma non posso esserne certo senza sapere esattamente cosa
stavi facendo prima di premere Annulla/Ripristina:

1. **Tasto [T] (aggancio automatico di una crepa importata)**: quando
   corregge una crepa, la modifica sul posto (stessa crepa, forma
   diversa) ma la registrava in Annulla come se fosse "una crepa nuova
   aggiunta". Risultato: premendo Annulla dopo un [T], invece di
   ripristinare la forma precedente, cancellava l'intera crepa. Corretto
   registrando la forma originale (posizione iniziale/finale/percorso)
   così Annulla ripristina davvero quella, senza toccare l'esistenza
   della crepa.

2. **Tasto [V] (ritraccia in blocco tutte le crepe importate)**: questa
   modifica anche più crepe insieme, ma non registrava assolutamente
   nulla per Annulla. Risultato: Annulla dopo un [V] non faceva niente,
   oppure annullava per sbaglio un'azione precedente non correlata.
   Corretto registrando la forma originale di ogni crepa toccata in
   un'unica voce, così un solo Annulla le ripristina tutte insieme
   (coerente col fatto che [V] è una singola azione dell'operatore).

Se il problema che hai visto **non** coinvolgeva [T] o [V] -- cioè hai
semplicemente disegnato una crepa, premuto Annulla, e Ripristina non
l'ha riportata indietro, senza altro nel mezzo -- allora questi due
correzioni non sono la causa e mi servirebbe sapere con più precisione
la sequenza esatta di tasti/click che hai usato, per continuare a
cercare nel punto giusto invece di continuare a tirare a indovinare.

**143 test in totale** ora, tutti verdi.

## Invio ancora rotto (causa vera trovata) + trascinamento reale sulla guida

Avevo frainteso entrambe le segnalazioni precedenti. Chiarito:

**"Invio non funziona" persisteva anche dopo la correzione sul focus dei
pulsanti** -- perché la causa reale era un'altra. La funzione che traduce
i tasti di Tkinter in codici per il programma si basava su `event.char`
anche per Invio/Backspace/Canc/Esc, assumendo che Tkinter lo popolasse in
modo affidabile con il carattere di controllo giusto (`'\r'`, ecc.) --
non garantito su tutte le piattaforme per tasti non stampabili, a
differenza di lettere e numeri normali. Corretto usando lo stesso
approccio già in uso per le frecce: questi tasti ora vengono riconosciuti
direttamente dal loro nome Tkinter (`keysym`), non più da `event.char`.
Aggiunti anche `KP_Enter` (invio del tastierino numerico). La correzione
precedente sul focus dei pulsanti resta comunque valida e utile, solo
non era la causa di *questo* problema.

**La "barra verticale che non funziona" era in realtà quella della guida
(`?`), non il canvas della foto** -- avevo frainteso a quale barra si
riferisse il messaggio precedente. Il click sulla barra della guida
faceva già qualcosa (saltava di una "pagina"), ma **trascinarla** non
faceva nulla: gestivo solo la pressione del pulsante del mouse, non il
movimento con il pulsante tenuto premuto. Ora è un trascinamento vero:
clic sulla barra salta subito a quel punto proporzionalmente E inizia il
trascinamento; muovere il mouse tenendo premuto continua a seguirlo
(anche oltre i bordi della barra stessa, con blocco ai limiti); rilasciare
il pulsante conclude il trascinamento.

**138 test in totale** ora, tutti verdi.

## Invio non funzionava per "Save and next" + barra orizzontale rimossa

**Invio non salvava**: causa trovata -- cliccare un pulsante della barra
strumenti gli dava il "focus" della tastiera in Tkinter, e i pulsanti
spesso intercettano Invio per conto proprio invece di lasciarlo passare
al resto del programma. Corretto in due modi: i pulsanti non prendono
più il focus quando cliccati (`takefocus=0`), e in più, dopo ogni click
su un pulsante o voce di menu, il focus torna esplicitamente al canvas
della foto -- coprendo entrambe le strade con cui il problema poteva
presentarsi.

**Barra di scorrimento orizzontale**: rimossa invece che corretta. Il
motivo: la larghezza minima della finestra è ora legata alla larghezza
della barra strumenti (per mostrarla per intero, vedi sotto), che con 13
pulsanti è già più larga della foto stessa (1200px). Questo significa
che il canvas non può MAI essere più stretto della foto -- la barra
orizzontale non aveva mai nulla da scorrere, quindi trascinarla non
faceva nulla non per un difetto di interazione, ma perché strutturalmente
non c'era mai spazio "oltre" da raggiungere. Tolta per non lasciare un
controllo che sembra rotto quando in realtà è solo superfluo.

**127 test in totale** ora, tutti verdi.

## Due correzioni importanti: Undo/Redo e scorrimento col mouse

**Redo non ripristinava una crepa annullata con Undo -- bug vero,
preesistente nel codice originale.** Trovato analizzando il codice: il
tasto U (Undo), quando annulla una crepa appena disegnata, la rimette in
coda per un eventuale Redo etichettandola `'crack_new'`. Il tasto R
(Redo) però controllava solo le etichette `'crack'`/`'crack_delete'` --
mai `'crack_new'` -- quindi quella specifica situazione (la più comune:
disegna una crepa, annullala, prova a ripristinarla) non veniva mai
riconosciuta: l'elemento veniva tolto dalla coda ma scartato in
silenzio, senza nessun errore visibile. Corretto aggiungendo
`'crack_new'` alle etichette riconosciute da Redo. Verificato con un
test che rifà l'intero percorso reale (disegna una crepa finta, Undo,
Redo) invece di limitarsi a controllare il pezzo di codice isolato --
proprio perché un test più superficiale non avrebbe scoperto il problema.

**Scorrimento verticale via mouse**: quello che percepivi come
"scorrimento che funziona da tastiera" era in realtà la funzione di
zoom/panoramica già esistente nell'app (tasto frecce), un meccanismo
completamente diverso dalla nuova barra di scorrimento Tkinter. Quella
vera aveva un problema reale: collegare la rotellina del mouse
direttamente al widget canvas è noto per essere inaffidabile tra
piattaforme diverse in Tkinter. Corretto usando lo schema più robusto
raccomandato proprio per questo problema: la rotellina si "aggancia" a
livello di finestra solo mentre il cursore è sopra il canvas (eventi
Enter/Leave), garantendo che l'evento venga sempre intercettato
indipendentemente da quale widget abbia il focus in quel momento.

**124 test in totale** ora, tutti verdi.

## Cinque aggiustamenti dopo il primo test funzionante

Lo screenshot che hai mandato confermava che l'impianto generale
funziona bene -- questi sono aggiustamenti mirati:

1. **Barra strumenti tagliata a destra (Info/Help non visibili)**: la
   larghezza della finestra ora viene misurata dalla larghezza *reale*
   richiesta dalla barra strumenti (`winfo_reqwidth()`), non da una
   formula fissa legata al canvas -- si allarga automaticamente
   qualunque sia il numero di pulsanti, entro i limiti dello schermo.

2. **Click sulla guida che attivava lo strumento crepe** (i quadratini
   gialli/arancioni): la barra di scorrimento della guida era solo
   disegnata (pixel sull'immagine), non un controllo Tkinter vero --
   cliccarci sopra passava il click al canvas sottostante, attivando lo
   strumento crepe. Corretto: quando la guida è aperta, `mouse_callback`
   ora intercetta ogni evento del mouse con priorità assoluta -- un
   click dentro l'area della barra di scorrimento scorre la lista (metà
   superiore = pagina su, metà inferiore = pagina giù), un click altrove
   viene semplicemente ignorato invece di raggiungere lo strumento.

3. **Carattere della barra di stato**: aumentato a 12pt (era 9pt), righe
   visibili ridotte da 5 a 2 come richiesto.

4. **Previous/Next disattivati ai limiti della coda**: "Previous" è
   disattivato sulla prima immagine, "Next" sull'ultima -- aggiornato a
   ogni fotogramma confrontando l'immagine corrente con la posizione
   nella coda (che resta sincronizzata anche dopo le rinomine
   automatiche, vedi la correzione precedente).

5. **"Session time" in alto a destra, poco contrasto**: aggiunto lo
   stesso pannello semi-trasparente "vetro" già usato per Info/Help
   dietro a quel testo.

**121 test in totale** ora, tutti verdi.

## Novità di questa consegna: scorrimento, navigazione senza salvare, guida "vetro"

**Finestra ora ridimensionabile, con barre di scorrimento sul canvas.**
Segnalazione ricorrente "la barra di stato non si vede": la causa vera
non era una misura sbagliata (già corretta due volte prima), ma
un'impostazione strutturale -- la finestra era bloccata a dimensione
fissa, e il canvas da solo riservava sempre 900px pieni. Ora:
- Il canvas vive dentro un contenitore con **barra di scorrimento
  verticale e orizzontale** -- il widget può essere più piccolo del
  contenuto (1200x900) e restare comunque completamente raggiungibile
  scorrendo.
- La finestra è **ridimensionabile** (prima era bloccata), si apre a
  un'altezza iniziale prudente pensata per stare su schermi comuni, e la
  puoi allargare per vedere più foto in una volta (fino a 1200x900 pieni
  su schermi grandi) o rimpicciolire.
- Barra strumenti e barra di stato sono ora "agganciate" per prime
  (sopra/sotto), e solo il canvas si adatta allo spazio che resta --
  quindi restano sempre visibili indipendentemente da quanto è alta la
  finestra, invece di dipendere da un calcolo preciso che due volte non
  ha funzionato.
- **Rotellina del mouse**: scorre la foto normalmente; se la guida (`?`)
  è aperta, scorre invece la sua lista di comandi.
- I click ora passano per `canvasx()`/`canvasy()` di Tkinter, che
  traducono correttamente la posizione anche quando la vista è scorsa --
  senza questo, un click su una vista scorsa sarebbe finito nel punto
  sbagliato.

**Effetto "vetro" esteso alla guida (`?`) + scorrimento.** La lista
comandi è più lunga dello spazio disponibile ed era tagliata in fondo
senza modo di raggiungere il resto. Ora:
- Stesso pannello semi-trasparente e stessa palette di colori
  ammorbiditi già usati per l'overlay Info (che avevi apprezzato).
- Solo una porzione della lista viene disegnata per volta, con una vera
  barra di scorrimento (traccia + cursore) disegnata sul bordo destro
  del pannello.
- **[Su]/[Giù]** (frecce, o rotellina del mouse) scorrono la lista
  mentre la guida è aperta -- fuori dalla guida, le frecce continuano a
  fare quello che hanno sempre fatto (panoramica/estensione crepa).

**Pulsanti "Previous Image" / "Next Image" (◀/▶) nella barra strumenti
e nel menu File.** Permettono di sfogliare le foto della cartella
**senza salvare** -- diversi da [S]/[Q]/[Invio], che salvano sempre.
Per rendere possibile tornare indietro ho dovuto riscrivere il ciclo
principale (`run()`) da un semplice "for" (solo avanti) a un contatore
manuale -- e proprio scrivendo questo ho trovato un bug reale
preesistente: quando una foto viene rinominata automaticamente
(assegnazione del gruppo edificio), l'elenco delle foto in coda non
veniva aggiornato con il nuovo nome. Tornando indietro con "Previous
Image" verso una foto già rinominata, il programma avrebbe cercato di
riaprire un file che non esisteva più con quel nome. Corretto e
verificato con un test end-to-end dedicato.

**Barra di stato**: corretto un punto debole reale trovato nel codice
(un aggiornamento fallito della barra poteva propagarsi e rompere
silenziosamente altra logica dell'app) -- vedi la sezione dedicata più
sotto per i dettagli, resta valida.

**110 test in totale** ora, tutti verdi.

## Correzione crash "CrackSegmentation quit unexpectedly" (importante)

Dal log del crash che mi hai mandato, la causa era chiara: su macOS
`cv2.waitKey()` non pompa solo la propria finestra, ma **l'intero ciclo
eventi nativo del processo**. Con una finestra Tkinter vera nello stesso
processo, un suo ridisegno che capita mentre `cv2.waitKey()` sta pompando
quel ciclo condiviso fa scattare il callback di Tk fuori dal contesto che
Python si aspetta -- corrompendo il thread state dell'interprete e
causando l'abort fatale (SIGABRT) che hai visto.

Ho trovato **ogni** chiamata a `cv2.waitKey()` nel codice (6 in totale,
non solo quella del loop principale -- due erano dentro finestre di
conferma "Salvataggio" e "Filtro crepe incompatibili", peraltro già
invisibili in modalità GUI perché disegnate su finestre cv2 separate mai
mostrate). Tutte e 6 estratte in metodi sovrascrivibili (stesso schema
degli altri hook -- comportamento da terminale identico), e in modalità
GUI **nessuna chiama più il vero `cv2.waitKey()`**: la tastiera arriva
già tramite Tkinter, e le due finestre di conferma sono diventate
dialoghi Tkinter veri (finalmente visibili, cosa che prima non erano).

Verificato con un test end-to-end dedicato che fa girare un'intera
sessione (compreso un salvataggio) e controlla esplicitamente che
`cv2.waitKey` non venga mai chiamato -- prova diretta che il
meccanismo del crash è eliminato, non solo "sembra funzionare".

Come precauzione aggiuntiva ho anche rimesso la finestra di console
(`console=True` nello spec PyInstaller, tolta nella consegna precedente):
non era la causa del crash, ma è la modifica più recente e meno
necessaria, e avere una console visibile aiuta a diagnosticare eventuali
problemi futuri.

## Icone nel menu

Ogni voce del menu -- inclusi i nomi di primo livello nella barra
(File, Edit, Tools...) -- ha ora un simbolo Unicode davanti al testo,
come icona leggera. Non sono immagini bitmap (macOS non le supporta nei
titoli di primo livello della barra menu di sistema, è un limite del
sistema operativo, non di Tkinter), ma caratteri di testo veri e propri:
nessun file esterno da includere nel pacchetto, nitidi a qualsiasi
risoluzione dello schermo. Stessa cosa per i pulsanti della barra
strumenti. Un test verifica che ogni voce abbia davvero un'icona, per
evitare regressioni.

## Come si usa

Identico a prima:
```
python3 crack_segmentation_gui.py
```
Si apre UNA finestra: menu in alto, una riga di pulsanti per le azioni
più comuni, la foto sotto. La tastiera funziona sempre, esattamente come
i tasti originali (S, W, L, +/-, frecce, ecc.) -- il menu/i pulsanti sono
solo un modo alternativo di mandare lo stesso comando.

Da riga di comando senza GUI, invariato:
```
python3 smart_segmentation.py
```

## Testo FILE/Cracks Length/BUILDING ora nascondibile (tasto J / pulsante Info)

Il testo verde e ciano sopra la foto (riga "FILE: ...", "Cracks Length:
...", "EDIFICIO"/"BUILDING: ...") veniva disegnato **sempre**, senza modo
di toglierlo. Corretto:

- **Di default ora è nascosto** (`show_info_overlay = False` in
  `config.py`) -- non più un overlay permanente.
- **Tasto [J]** (libero, nessuna collisione) lo mostra/nasconde. Stessa
  cosa dal pulsante **ℹ Info** aggiunto nella barra strumenti (icona
  circolare con la "i", la stessa che mi hai allegato) e dalla voce
  "Toggle Info" nel menu Visualizza -- premi una volta per mostrarlo,
  premi di nuovo per nasconderlo, esattamente come richiesto.
- **Colori ammorbiditi**: un pannello scuro semi-trasparente dietro al
  testo (stesso stile già usato per la guida a schermo `?`), con testo
  bianco-grigio tenue e oro/crema invece del verde e ciano acceso di
  prima. La riga BUILDING mantiene il significato dei colori (verde =
  assegnato, arancio = da assegnare, rosso = errore) ma con toni più
  tenui, non più puro RGB acceso.

## Barra di stato: correzione difensiva

Segnalazione "la barra di stato non funziona": ho trovato un punto
debole concreto nel codice, anche se non ho potuto riprodurre il
problema esatto in questo ambiente (nessun display reale qui). Il punto
che aggiorna la barra con ogni nuovo messaggio non aveva protezione: se
per qualsiasi motivo quell'aggiornamento avesse sollevato un'eccezione,
l'errore si sarebbe propagato fino al `print()` che l'aveva innescato --
e dato che gran parte di `smart_segmentation.py` ha blocchi
`except Exception` piuttosto ampi, un singolo aggiornamento fallito della
barra avrebbe potuto far sparire silenziosamente anche logica applicativa
non correlata, non solo il testo nella barra stessa. Ora è protetto: un
problema nella barra di stato non può più propagarsi altrove. Aggiunto un
test dedicato che simula esattamente questo scenario.

Se il problema persiste anche dopo questa correzione, fammi sapere
esattamente cosa vedi (vuota? pulsanti che non rispondono? testo
tagliato?) -- mi aiuta a restringere il campo rispetto a tirare a
indovinare.

## Tutto in inglese (menu, barra strumenti, guida a schermo, console)

**Traduzione completa** in questa consegna -- prima era tradotta solo la
GUI che avevo scritto io (menu, barra strumenti, finestre di dialogo),
mentre il testo HUD sopra la foto e i ~160 `print()`/messaggi a schermo
nel codice principale (`smart_segmentation.py` e
`compare_and_filter_cracks.py`) restavano in italiano per una scelta di
progetto di una consegna precedente. Ora è tutto in inglese, **inclusa la
guida comandi a schermo** (tasto `?`), che era rimasta indietro nel primo
giro di traduzione.

Cosa NON ho tradotto, di proposito, perché non è testo mostrato
all'operatore:
- Due citazioni testuali di segnalazioni utente dentro i commenti del
  codice (`"quando carico il programma o passo all'immagine
  successiva"`, `"schermo bianco... non funziona"`) -- sono preservate
  parola per parola come prova storica di cosa fu effettivamente
  segnalato, esattamente come già avveniva altrove nel progetto.
- Il doppio riconoscimento `"crack"/"crepa"` e `"detachment"/"distacco"`
  quando il programma LEGGE un JSON esistente -- se lo traducessi
  rischierei di rompere la compatibilità con file esportati da versioni
  precedenti dello strumento che potrebbero ancora avere l'etichetta
  italiana.
- Nomi di variabili Python interne (es. `nome_foto_corrente`) -- non
  sono testo visibile, rinominarle non cambia nulla per chi usa il
  programma ma aggiunge rischio inutile.

99 test ancora tutti verdi dopo la traduzione (nessun test dipendeva dal
testo esatto dei messaggi, tranne uno che ho aggiornato di conseguenza).

## Inglese nell'interfaccia + striscia di stato incorporata

**Testo tradotto in inglese**: menu, barra strumenti, finestre di
dialogo (scelta modalità, calibrazione, errori) -- tutto ciò che questa
GUI genera.

**Striscia di stato fissa, incorporata in basso** nella finestra
principale (non più una finestra separata -- tornata indietro su
segnalazione: c'era in realtà spazio libero sullo schermo, il problema
vero non era la mancanza di spazio ma il modo in cui dimensionavo la
finestra). Prima mi affidavo solo al ridimensionamento automatico di
Tkinter (calcola la dimensione "naturale" dal contenuto, poi blocca);
adesso invece la finestra viene dimensionata **esplicitamente**, con
larghezza e altezza dichiarate in pixel calcolate dopo aver costruito
tutto il contenuto (menu, barra strumenti, canvas, striscia di stato),
invece di sperare che il blocco implicito la calcoli giusta da solo.

Ha una vera barra di scorrimento verticale oltre ai due pulsanti freccia
(▲/▼) richiesti. Cattura automaticamente tutto ciò che il programma
stampa in console -- compresi i messaggi italiani sopra citati -- senza
aver toccato un solo `print()`: ho semplicemente "agganciato"
`sys.stdout`/`sys.stderr` così che ogni scrittura arrivi sia alla
console vera (se c'è) sia alla striscia. Risolve in modo generale lo
stesso problema per cui avevo già aggiunto un popup dedicato per
l'errore "cartella Images vuota" -- ora QUALSIASI messaggio è visibile
anche quando lanci l'app col doppio click (senza terminale collegato).

**La console era stata tolta dal pacchetto in una consegna precedente**
(`console=False`, sostituita dalla striscia di stato) ma è stata rimessa
(`console=True`) come precauzione dopo il crash -- vedi la sezione
dedicata più sopra. `packaging/crack_segmentation_gui.spec` spiega come
toglierla di nuovo quando si sarà sicuri che non serva più per il
debug.

**99 test in totale** ora, incluso uno specifico che verifica che la
finestra riceva davvero una dimensione esplicita (larghezza x altezza),
non solo una posizione, per evitare che questo stesso problema si
ripresenti in futuro senza che un test lo segnali.

## Perché è ancora sicuro (nessuna riscrittura della logica)

Tre cose bastano a mostrare un'immagine e farci lavorare sopra: disegnare
i pixel, leggere i click del mouse, leggere i tasti. Per ciascuna ho
trovato il punto giusto per agganciarmi senza toccare la logica:

**Disegno.** `render_scene()` (invariato) produce sempre lo stesso
fotogramma di sempre -- ogni crepa, marcatore, testo HUD -- esattamente
come prima. L'UNICA differenza è che ora, invece di chiamare
direttamente `cv2.imshow(...)`, chiama `self._display_frame(...)`: un
nuovo aggancio che di default fa esattamente `cv2.imshow(...)` (nessun
cambiamento da riga di comando), e che la finestra unica sovrascrive per
disegnare invece nel proprio canvas incorporato.

**Mouse.** `mouse_callback(event, x, y, flags, param)` (invariato) è
stato letto con attenzione prima di collegarci qualcosa: usa solo tre
eventi (click, rilascio, movimento) e non legge mai `flags`/`param`. I
click sul canvas Tkinter vengono quindi passati a QUESTO STESSO metodo,
con gli stessi valori che una finestra OpenCV vera gli avrebbe mandato --
zero traduzioni rischiose, zero modifiche a `mouse_callback` stesso.

**Coordinate.** `transform_window_to_real_coords()` /
`transform_real_to_window_coords()` (invariate) davano per scontato uno
spazio di visualizzazione di 1200x900 pixel -- un'assunzione già presente
in tutto `render_scene()` (posizioni HUD, controlli sui marcatori...).
Invece di reinventare quella matematica per farla combaciare con una
finestra di dimensione variabile (un rischio concreto: coordinate storte
= click nel punto sbagliato), il canvas incorporato è semplicemente
**costruito esattamente a 1200x900**, così quella matematica resta
corretta automaticamente, senza toccarla.

**Tastiera.** Riusa la stessa coda `_pending_keys`/`process_keypress()`
già presente e testata dalla consegna precedente (lì usata dai pulsanti
del vecchio pannello) -- un tasto premuto nella finestra Tkinter viene
tradotto nello stesso codice che `cv2.waitKey()` avrebbe restituito (vedi
`tk_key_event_to_code()`) e messo nella stessa coda.

## Perché esiste ancora una finestra OpenCV, anche se invisibile

Ho scoperto (con un test dedicato, prima di scrivere qualsiasi codice)
che diverse funzioni del programma chiamano direttamente
`cv2.getWindowImageRect("Crack Detector Workspace")` per sapere quanto è
grande la finestra -- e se quella finestra non esiste proprio, la
chiamata solleva un'eccezione che interrompe `render_scene()` **prima
ancora di disegnare qualsiasi cosa**, fotogramma completamente perso.

Per questo la finestra unica crea comunque una finestra OpenCV vera.
**Correzione dopo una segnalazione su un Mac reale**: inizialmente
provavo a tenerla nascosta spostandola fuori schermo
(`cv2.moveWindow(..., -10000, -10000)`), ma il window manager di macOS
l'ha rimessa dentro lo schermo, visibile per intero, esattamente quello
che si vedeva nello screenshot condiviso. Invece di rincorrere trucchi
specifici per ogni sistema operativo, ora **la correttezza non dipende
più da dove/come quella finestra finisce per davvero**:
`cv2.getWindowImageRect()` viene intercettata (solo per il nome "Crack
Detector Workspace") per restituire sempre la dimensione fissa del
canvas, qualunque cosa faccia realmente la finestra nascosta. Quella
finestra viene comunque rimpicciolita al minimo e spostata fuori schermo
come tentativo, ma ora è solo un dettaglio estetico -- se il sistema
operativo la mostra comunque da qualche parte, resta un fastidio visivo,
non più un bug funzionale.

## Limiti noti di questa prima versione

- **Il tasto [F] (schermo intero) non è stato verificato su un Mac o PC
  Windows reale.** Prova ancora ad agire sulla finestra OpenCV nascosta;
  dato il punto sopra, anche se dovesse renderla visibile non può più
  rompere il posizionamento di mouse/HUD (la correttezza non dipende più
  dal suo stato reale) -- nel peggiore dei casi resta solo una finestra
  di troppo a schermo, non un malfunzionamento. Se càpita, segnalamelo:
  la soluzione pulita sarebbe far agire F sulla finestra Tkinter vera
  (ingrandire/ripristinare) invece che su quella nascosta.

## Correzioni dopo il primo test su un Mac reale

Il primo screenshot inviato ha mostrato due problemi, entrambi corretti
in questa consegna:

1. **Finestra dell'app troppo piccola** (pulsanti tagliati, immagine
   minuscola invece che 1200x900). Causa: `resizable(False, False)`
   veniva chiamato PRIMA di creare menu/barra strumenti/canvas, quando la
   finestra era ancora vuota -- bloccandola a quella dimensione
   placeholder minuscola invece di lasciarla crescere fino al contenuto
   vero. Corretto spostando quella chiamata alla fine, dopo aver
   costruito tutto e chiesto a Tkinter di ricalcolare la dimensione reale
   (`update_idletasks()`).
2. **La finestra OpenCV nascosta era visibile** invece di restare fuori
   schermo -- vedi la sezione sopra ("Perché esiste ancora una finestra
   OpenCV") per la correzione.

Aggiunti 2 test dedicati per entrambi.

## Nuovo comando: nascondi/mostra il riempimento blu delle crepe (tasto N)

Il riempimento blu delle crepe tracciate veniva sempre disegnato senza
possibilità di nasconderlo (a differenza dei marcatori gialli/rossi, già
nascondibili con Spazio). Aggiunto il tasto **N** (libero, nessuna
collisione con nulla): alterna `cfg.show_crack_overlay`, di default
`True` -- quindi finché non lo premi, tutto appare come sempre. È anche
nel menu Visualizza e nella guida a schermo (`?`). I marcatori
gialli/rossi e il riempimento verde dei distacchi NON sono toccati da
questo tasto.

## Errore "silenzioso" al doppio click (corretto in una consegna precedente)

Lanciando l'app col doppio click su Mac (o l'.exe senza terminale su
Windows), Finder/Explorer non collegano mai una finestra di console.
Se ad esempio la cartella `Images` risultava vuota, l'app si chiudeva
subito dopo, senza che l'operatore vedesse alcun messaggio. Aggiunto un
hook (`_report_fatal_error`, stesso schema degli altri: no-op da riga di
comando) che ora fa apparire una finestra di errore vera in quel caso.

## Dove mettere le immagini da segmentare

**Da sorgente** (`python3 smart_segmentation.py` o `python3 crack_segmentation_gui.py`):
crea una cartella `Images` nella STESSA cartella dove si trovano
`config.py`/`smart_segmentation.py`, e mettici dentro le foto (jpg, jpeg,
png, bmp, tiff). Gli output finiscono in `segmentated images` e `already
processed images`, sempre lì accanto.

**Dalla app pacchettizzata (.app/.exe)**: `~/Documents/CrackSegmentation/Images`
(creata automaticamente al primo avvio) -- un `.app`/`.exe` pacchettizzato
non gira più "dentro" la sua cartella sorgente, quindi il programma usa
un posto normale e scrivibile del profilo utente invece che una cartella
sepolta dentro il pacchetto stesso.

## Pacchettizzazione in .dmg / installer Windows / pacchetto .deb Linux

**Importante**: né PyInstaller né gli strumenti di firma/pacchettizzazione
di macOS (`hdiutil`), Windows (Inno Setup) o Linux (`dpkg-deb`) fanno
compilazione incrociata. Ogni pacchetto va generato UNA VOLTA sul sistema
operativo reale a cui è destinato -- non è possibile generare quello di
un sistema da un altro.

Su Mac (produce sia `.app` che `.dmg` in un solo comando):
```
cd packaging
chmod +x build_mac.sh
./build_mac.sh
# risultato: packaging/dist/CrackSegmentation.app
#            packaging/dist/CrackSegmentation.dmg   <- da distribuire
```
Il `.dmg` si apre come un installer classico: doppio click, trascina
l'icona nella cartella Applications mostrata accanto, espelli il disco.

Su Windows (produce l'eseguibile, e un vero `Setup.exe` se Inno Setup è
installato):
```
cd packaging
build_windows.bat
REM risultato: packaging\dist\CrackSegmentation\CrackSegmentation.exe
REM            packaging\dist\CrackSegmentation-Setup.exe   <- se trova Inno Setup
```
Inno Setup (gratuito, https://jrsoftware.org/isdl.php) va installato una
volta sul PC Windows usato per compilare; lo script lo rileva da solo. Se
non è presente, lo script te lo segnala e si ferma comunque con un
eseguibile funzionante in `dist\CrackSegmentation\`.

Su Linux Ubuntu/Debian (produce l'eseguibile, e un pacchetto `.deb`
installabile con `apt`):
```
cd packaging
chmod +x build_linux.sh
./build_linux.sh
# risultato: packaging/dist/CrackSegmentation/CrackSegmentation
#            packaging/dist/cracksegmentation_1.0.0_amd64.deb   <- da distribuire
```
A differenza di Mac e Windows, Python su Linux **non include Tkinter di
default** -- va installato a parte col pacchetto di sistema
`sudo apt install python3-tk` prima di compilare (lo script lo controlla
e te lo segnala se manca). Lo script crea inoltre un ambiente virtuale
isolato (`.venv`) per installare le dipendenze, invece di toccare il
Python di sistema -- evita conflitti come `ImportError: numpy.core.
multiarray failed to import`, che capita quando un `matplotlib` installato
via `apt` resta incompatibile con la versione di `numpy` installata via
`pip` nello stesso Python di sistema (bug reale riscontrato su Ubuntu).
Il `.deb` si installa con
`sudo apt install ./cracksegmentation_1.0.0_amd64.deb`, aggiunge una voce
"Crack Segmentation" al menu Applicazioni e un comando `cracksegmentation`
lanciabile da terminale. Ho verificato la logica di generazione del `.deb`
in questo ambiente con un build PyInstaller fittizio (`dpkg-deb` è
disponibile qui) -- struttura del pacchetto, nome conforme alle
convenzioni Debian (minuscolo), permessi e voce da menu tutti corretti;
resta comunque da provare un avvio vero su una macchina Ubuntu reale,
come per Mac e Windows.

Tutti e tre gli script installano le dipendenze da `requirements.txt`
(inclusi Pillow e PyInstaller) e puliscono le build precedenti prima di
ricreare. Note pratiche incluse negli script stessi: Gatekeeper su Mac
(primo avvio va sbloccato manualmente), falsi positivi
antivirus/SmartScreen su Windows, e pacchetto non firmato su Linux
(tutti normali per un pacchetto locale non firmato, non un segno di
problemi).

**Non ho potuto eseguire una build PyInstaller reale né aprire una vera
finestra in questo ambiente** (sandbox Linux senza rete, senza Tkinter
installato, senza display): tutta la logica di traduzione mouse/tastiera
e il cablaggio di menu/pulsanti sono stati verificati con un doppio
livello di test (stub che registra le chiamate senza un vero Tk, più un
test end-to-end che fa girare l'intera sessione compreso il salvataggio),
e la logica di pacchettizzazione `.deb` è stata verificata con un build
fittizio -- ma il primo vero avvio va comunque provato una volta sui
vostri computer, su tutti e tre i sistemi operativi.

## Foto e pagine PDF centrate nel pannello

Sia la foto principale (canvas) sia le pagine del manuale PDF (vedi sezione
sopra) ora vengono **centrate orizzontalmente/verticalmente** nel loro
riquadro quando la finestra è più larga/alta del contenuto fisso (1200x900
per la foto), invece di restare ancorate nell'angolo in alto a sinistra
con un bordo nero esteso sull'altro lato.

Per la foto questo tocca un punto delicato: i clic del mouse devono
continuare a mappare esattamente sullo stesso pixel della foto reale
indipendentemente da quanto viene spostata per restare centrata --
`_canvas_xy()` ora sottrae lo stesso scostamento usato per posizionare
l'immagine, prima di passare le coordinate a `mouse_callback()`. Verificato
con 4 nuovi test dedicati proprio a questa corrispondenza clic-pixel.

## Manuale utente PDF al posto della guida a schermo (`?`)

Il tasto `?` (e la voce "Show user manual" nel menu/barra strumenti) ora
apre il **manuale utente PDF vero** (`Crack_Segmentation_User_Manual.pdf`,
incluso nel pacchetto) direttamente dentro la finestra principale, al
posto del vecchio testo sovraimpresso sul canvas. Il pannello ha
scorrimento **sia verticale che orizzontale** (una pagina PDF renderizzata
può essere più larga del riquadro, a differenza della foto che è sempre
1200x900): rotellina del mouse per scorrere verticalmente, trascinamento
delle barre per entrambe le direzioni.

Il rendering usa **PyMuPDF** (`pip install pymupdf`, nuova dipendenza in
`requirements.txt` e inclusa nello script di packaging come risorsa
bundle) invece di `pdf2image`/poppler: quest'ultimo richiederebbe un
binario di sistema esterno che PyInstaller non include automaticamente,
rompendo silenziosamente l'eseguibile pacchettizzato su una macchina
senza poppler installato. Il rendering è **pigro** (avviene solo al primo
utilizzo, non all'avvio dell'app) e non blocca il resto dell'app se
`pymupdf` non è installato o il PDF manca/è corrotto: mostra un messaggio
esplicativo nel pannello stesso invece di far crashare tutto.

Questo cambiamento riguarda **solo la modalità GUI**
(`crack_segmentation_gui.py`): la vecchia guida testuale sovraimpressa
resta identica e invariata in modalità senza interfaccia (avvio diretto
di `smart_segmentation.py`, o dello script `compare_and_filter_cracks.py`
da riga di comando), dove non esiste un pannello Tkinter in cui mostrare
un PDF. Tecnicamente: `show_help_menu` in `config.py` resta sempre
`False` durante l'uso della GUI (la sovrascrittura di `_toggle_help_menu`
in `GuiCrackSegmentation` non lo tocca più), quindi il vecchio codice di
disegno/scorrimento della guida in `render_scene()` semplicemente non si
attiva mai in quella modalità -- nessuna riga toccata lì.

**Non ho potuto installare `pymupdf` in questo ambiente** (sandbox senza
accesso a internet), quindi non ho potuto verificare visivamente il
rendering vero del PDF. La logica di integrazione (costruzione del
pannello, doppia barra di scorrimento, rendering pigro, gestione degli
errori, collegamento del tasto `?`) è verificata con 9 nuovi test che
usano un modulo `fitz` finto (`tests_support/fake_fitz.py`, stesso
principio di `fake_tkinter.py`: tutta la pipeline PIL/Tkinter gira per
davvero, solo la decodifica del PDF è simulata) -- ma il rendering vero
va comunque controllato su una macchina reale con `pymupdf` installato.

## Eseguire i test

```
python3 -m unittest discover -p "test_*.py" -v
```
