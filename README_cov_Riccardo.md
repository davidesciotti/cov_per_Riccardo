# 3x2pt covariance

Nella repo trovi il codice e le covarianze per la 3x2pt. Qui sotto un po' di chiarimenti sulle convenzioni, che devono coerenti per datavector e covariance matrix (altrimenti il $\chi^2$ ti restituisce valori senza senso):
* `probe_ordering`: l'ordinamento dei probe, per l'appunto. Ho assunto lo stacking (LL, GL, GG), che è quello "di default"
* `GL_or_LG`: se usare $C^{GL}_{ij}(\ell)$ o $C^{LG}_{ij}(\ell)$ (ricorda che vale $C^{LG}_{ij}(\ell) = C^{GL}_{ji}(\ell)$, i.e. basta trasporre gli indici di redshift). Anche qui, ultimamente stiamo usando GL come default, ma in caso tu voglia cambiare puoi semplicemente modificare questa stringa.
* `triu_tril`, `row_col_major`: per gli auto-spectra (LL e GG), che sono simmetrici in $i, j$, scegli se prendere prendendo solo la parte triangolare superiore o inferior ("upper/lower triangle, `triu`/`tril`") riga per riga o colonna per colonna (ovvero con un [ordinamento](https://en.wikipedia.org/wiki/Row-_and_column-major_order) `row-major` o `col-major`).
* `block_index`: nella matrice di covarianza 2D (che, alla fine, è l'unico file che ti interessa; gli altri li ho inclusi per completezza), i blocchi sulla diagonale corrispondono a una coppia $(\ell_1, \ell_2)$. Il fatto che la matrice sia block-diagonal deriva dal fatto che, per la parte Gaussiana, non c'è covarianza per $\ell_1 \neq \ell_2$. L'alternativa al valore `ell` è `zpair`, che significa che i blocchi sono indicizzati dalle coppie di redshift: $(zpair_i, zpair_j)$; il discorso è lo stesso ma è un po' meno intuitivo. Comunque, per farti un'idea di com'è fatta la matrice 2D, puoi usare `plt.matshow(cov_2D)`, magari dandogli il log della covarianza 

Nota che salvo i file in formato `npz`. Per caricarli, usa

    cov = np.load(filepath)['arr_0']

