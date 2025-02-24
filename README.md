# best_prompt
alcolo delle Similarità

    Il modello SentenceTransformer codifica il nuovo prompt in un embedding vettoriale.
    Questo embedding viene confrontato con quelli dei prompt esistenti utilizzando il coseno di similarità (util.pytorch_cos_sim).
    Il risultato è un array di similarità, in cui ogni valore rappresenta quanto il nuovo prompt è simile ai prompt nel dataset.

Ponderazione dei Punteggi

    Ogni prompt esistente ha un punteggio associato nel dataset.
    I punteggi vengono moltiplicati per le relative similarità, ottenendo un array di punteggi ponderati.

weighted_scores=scores×similarities
weighted_scores=scores×similarities

Calcolo del Punteggio Finale

    Il punteggio finale viene ottenuto facendo la somma dei punteggi ponderati e normalizzandoli con la somma delle similarità:

final_score=∑(scores×similarities)∑similarities
final_score=∑similarities∑(scores×similarities)​

Questo assicura che i prompt con maggiore similarità abbiano più peso nel determinare il punteggio finale.

Selezione della Miglior Corrispondenza

    Il prompt con la similarità più alta viene identificato e il valore della similarità massima viene riportato.

Salvataggio nei Log

    Ogni operazione viene registrata in log.txt, includendo il prompt, il punteggio assegnato e la similarità migliore.