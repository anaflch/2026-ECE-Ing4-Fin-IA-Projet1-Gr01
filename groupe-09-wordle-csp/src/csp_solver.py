from collections import Counter

def wordle_feedback_vjg(secret: str, guess: str) -> str:
    """
    Feedback Wordle (FR) :
      V = vert, J = jaune, G = gris
    Règles Wordle exactes, y compris doublons.
    """
    secret = secret.strip().upper()
    guess = guess.strip().upper()
    if len(secret) != 5 or len(guess) != 5:
        raise ValueError("secret et guess doivent faire 5 lettres")

    res = ["G"] * 5
    remaining = Counter(secret)

    # 1) Verts
    for i in range(5):
        if guess[i] == secret[i]:
            res[i] = "V"
            remaining[guess[i]] -= 1

    # 2) Jaunes
    for i in range(5):
        if res[i] == "V":
            continue
        ch = guess[i]
        if remaining[ch] > 0:
            res[i] = "J"
            remaining[ch] -= 1

    return "".join(res)


def solve_wordle_csp(possible_words, attempts):
    """
    Résout Wordle par filtrage de contraintes (approche CSP "par vérification").

    Conceptuellement :
      - Variables : le mot secret (un mot de 5 lettres)
      - Domaine   : possible_words (ton dictionnaire de mots 5 lettres)
      - Contraintes : chaque (guess, feedback) impose une contrainte sur le secret :
            feedback(secret, guess) == feedback_observé

    Paramètres
    ----------
    possible_words : iterable[str]
        Liste / set / générateur de candidats (mots potentiellement secrets).
    attempts : list[tuple[str, str]]
        Liste des tentatives sous la forme [(guess, feedback), ...]
        - guess : mot proposé (5 lettres)
        - feedback : chaîne de 5 caractères dans {V, J, G}
            V = vert, J = jaune, G = gris

    Retour
    ------
    list[str]
        Tous les mots de possible_words compatibles avec TOUTES les contraintes.
    """

    # -------------------------------------------------------------------------
    # 1) Nettoyage / validation des contraintes (attempts)
    #    Objectif : ignorer toute entrée mal formée plutôt que planter le solver.
    # -------------------------------------------------------------------------
    cleaned_attempts = []

    for item in attempts:
        # On attend un couple (guess, fb). Si ce n'est pas le cas : on ignore.
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue

        guess, fb = item

        # Les deux éléments doivent être des strings
        if not isinstance(guess, str) or not isinstance(fb, str):
            continue

        # Normalisation (le solver travaille en majuscules, sans espaces)
        guess = guess.strip().upper()
        fb = fb.strip().upper()

        # Wordle = 5 lettres, 5 feedbacks
        if len(guess) != 5 or len(fb) != 5:
            continue

        # Feedback doit contenir uniquement V/J/G
        if any(c not in "VJG" for c in fb):
            continue

        # Guess doit contenir uniquement des lettres A-Z
        # (si tu gères les accents, on pourra adapter plus tard)
        if any(not ("A" <= ch <= "Z") for ch in guess):
            continue

        cleaned_attempts.append((guess, fb))

    # -------------------------------------------------------------------------
    # 2) Filtrage du dictionnaire
    #    Pour chaque mot candidat w, on vérifie toutes les contraintes :
    #        wordle_feedback_vjg(w, guess) doit être EXACTEMENT fb
    #    Si une contrainte échoue, w est éliminé.
    # -------------------------------------------------------------------------
    solutions = []

    for w in possible_words:
        # Normalisation du candidat
        w = w.strip().upper()

        # On ignore ce qui n'a pas exactement 5 lettres
        if len(w) != 5:
            continue

        # Hypothèse : w est valide tant qu'aucune contrainte ne le contredit
        ok = True

        for guess, fb in cleaned_attempts:
            # Contrainte principale (modèle Wordle exact, y compris doublons)
            if wordle_feedback_vjg(w, guess) != fb:
                ok = False
                break  # inutile de tester les autres contraintes

        if ok:
            solutions.append(w)

    return solutions

