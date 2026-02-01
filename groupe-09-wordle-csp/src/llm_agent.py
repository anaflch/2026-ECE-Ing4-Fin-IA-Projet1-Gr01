import json
import re
from typing import Optional

import ollama

from csp_solver import solve_wordle_csp


# ---------------------------------------------------------------------------
# Dictionary loader
# ---------------------------------------------------------------------------
def load_dictionary(filename: str) -> list[str]:
    """
    Charge un dictionnaire de mots depuis un fichier texte.

    Retourne :
      - une liste de mots (str) en uppercase, longueur 5
      - [] si le fichier n'est pas trouvé
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            # On filtre sur len(line.strip()) == 5 AVANT upper() : équivalent ici.
            return [line.strip().upper() for line in f if len(line.strip()) == 5]
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return []


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def normalize_guess(s: str) -> Optional[str]:
    """
    Valide + normalise un guess Wordle.

    Règles :
      - doit être une str
      - strip + uppercase
      - longueur 5
      - uniquement A-Z (pas d'accents, pas de tirets)

    Retour :
      - le guess normalisé (str) si OK
      - None sinon
    """
    if not isinstance(s, str):
        return None

    s = s.strip().upper()

    if len(s) != 5:
        return None

    # Contrôle strict A-Z (utile pour éviter des caractères invisibles / accents)
    if any(not ("A" <= ch <= "Z") for ch in s):
        return None

    return s


def normalize_feedback(s: str) -> Optional[str]:
    """
    Valide + normalise un feedback Wordle au format V/J/G.

    Règles :
      - doit être une str
      - strip + uppercase
      - longueur 5
      - uniquement dans {V, J, G}

    Retour :
      - feedback normalisé (str) si OK
      - None sinon
    """
    if not isinstance(s, str):
        return None

    s = s.strip().upper()

    if len(s) != 5:
        return None

    if any(c not in "VJG" for c in s):
        return None

    return s


# ---------------------------------------------------------------------------
# Parsing direct (sans LLM)
# ---------------------------------------------------------------------------
# Formats acceptés :
#   "ORATE GVVJG"
#   "ORATE->GVVJG"
#   "ORATE -> GVVJG"
_DIRECT = re.compile(r"^\s*([A-Za-z]{5})\s*(?:->\s*)?([VvJjGg]{5})\s*$")


# ---------------------------------------------------------------------------
# LLM extraction (fallback)
# ---------------------------------------------------------------------------
def extract_attempt_from_text(user_text: str) -> Optional[dict]:
    """
    Utilise le LLM pour extraire EXACTEMENT une tentative Wordle depuis du texte libre.

    Objectif :
      - l'utilisateur peut écrire : "j'ai joué orate et j'ai eu g v v j g"
      - on veut récupérer un couple (guess, feedback) strict

    Stratégie :
      - on force le modèle à répondre via un "tool call" (fonction extract_wordle_attempt)
      - on revalide ensuite côté Python 

    Retour :
      - {"guess": "ORATE", "feedback": "GVVJG"} si extraction OK
      - None sinon
    """
    response = ollama.chat(
        model="llama3.1",
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a Wordle assistant.\n"
                    "Task: extract EXACTLY ONE Wordle attempt from the user's text.\n\n"
                    "You MUST return ONLY a tool call to extract_wordle_attempt.\n"
                    "Rules:\n"
                    "- guess: a 5-letter ENGLISH word (A-Z)\n"
                    "- feedback: a 5-character string using ONLY V, J, G\n"
                    "  V=green, J=yellow, G=gray\n"
                    "- Do NOT invent data: if guess or feedback is missing/unclear, "
                    'return guess="" and feedback="".\n\n'
                    f"USER TEXT:\n{user_text}"
                ),
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "extract_wordle_attempt",
                    "description": "Extract one Wordle attempt (guess + V/J/G feedback)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "guess": {"type": "string"},
                            "feedback": {"type": "string"},
                        },
                        "required": ["guess", "feedback"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    )

    # Ollama renvoie typiquement un dict avec "message"
    msg = response.get("message", {}) or {}
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        return None

    # arguments peut être déjà un dict ou une string JSON (selon version / config)
    args = tool_calls[0]["function"]["arguments"]

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None

    if not isinstance(args, dict):
        return None

    # Re-validation côté code : sécurité + robustesse.
    guess = normalize_guess(args.get("guess", ""))
    feedback = normalize_feedback(args.get("feedback", ""))

    if not guess or not feedback:
        return None

    return {"guess": guess, "feedback": feedback}


# ---------------------------------------------------------------------------
# Full agent (CSP + LLM ranking)
# ---------------------------------------------------------------------------
MAX_CANDIDATES_TO_LLM = 40


def interroger_agent_wordle(prompt_utilisateur: str, dictionary_words, attempts: list):
    """
    Pipeline complet de l'agent Wordle.

    Entrées :
      - prompt_utilisateur : texte brut utilisateur (ex: "ORATE -> GVVJG" ou phrase libre)
      - dictionary_words : liste de mots 5 lettres (domaine CSP)
      - attempts : historique MUTABLE des tentatives [(guess, feedback), ...]
                  (persisté entre tours côté Streamlit/session_state)

    Étapes :
      1) parse direct via regex (rapide, déterministe)
      2) fallback extraction via LLM si le texte est libre
      3) append dans l'historique
      4) CSP: filtrage des candidats compatibles
      5) si trop de candidats, on sous-échantillonne ceux envoyés au LLM (coût/latence)
      6) LLM: propose un ranking / next guess parmi les candidats
    """

    # 1) Parsing direct : si l'utilisateur donne un format structuré, pas besoin de LLM
    m = _DIRECT.match(prompt_utilisateur or "")
    if m:
        guess = m.group(1).upper()
        feedback = m.group(2).upper()
    else:
        # 2) Fallback : extraction sémantique via LLM (cas "texte libre")
        extracted = extract_attempt_from_text(prompt_utilisateur)
        if not extracted:
            return (
                "Could not extract a valid attempt.\n"
                "Expected format: 'ORATE GVVJG' or 'ORATE -> GVVJG' "
                "(V=green, J=yellow, G=gray)."
            )
        guess = extracted["guess"]
        feedback = extracted["feedback"]

    # Optionnel mais utile : revalider même après regex (cohérence + sécurité)
    guess = normalize_guess(guess)
    feedback = normalize_feedback(feedback)
    if not guess or not feedback:
        return "Invalid guess/feedback after normalization. Please use 5 letters and V/J/G."

    # 3) Mise à jour de l'historique des contraintes
    attempts.append((guess, feedback))

    # 4) CSP solving = filtrage du domaine par toutes les contraintes collectées
    possible = solve_wordle_csp(dictionary_words, attempts)

    # Si plus aucun mot ne satisfait les contraintes, il y a incohérence (erreur feedback,
    # mot hors dictionnaire, ou extraction incorrecte)
    if not possible:
        return (
            "No solution matches the current constraints.\n"
            f"Last attempt: {guess} -> {feedback}\n"
            f"History: {attempts}"
        )

    # 5) On limite le nombre de candidats envoyés au LLM (latence + coût)
    candidates_for_llm = possible[:]

    if len(candidates_for_llm) > MAX_CANDIDATES_TO_LLM:
        # Heuristique simple : favoriser les mots avec plus de lettres distinctes
        # (souvent bon pour "explorer" en Wordle)
        candidates_for_llm = sorted(
            candidates_for_llm, key=lambda w: len(set(w)), reverse=True
        )[:MAX_CANDIDATES_TO_LLM]

    # 6) LLM ranking : on lui donne la liste, et on lui interdit d'inventer
    prompt_final = f"""
You are an expert Wordle solver.

You are given a list of valid 5-letter ENGLISH words.
You MUST choose words ONLY from this list.
It is strictly forbidden to invent or modify any word.

List of possible words:
{candidates_for_llm}

Return STRICTLY:

Chosen word: <WORD>

Priority ranking:

1. <WORD>
2. <WORD>
3. <WORD>
"""

    final_response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt_final}],
    )

    content = final_response["message"]["content"]

    # Affichage "humain" : on montre un extrait des candidats CSP
    shown = ", ".join(possible[:30]) + ("..." if len(possible) > 30 else "")

    note = ""
    if len(possible) > MAX_CANDIDATES_TO_LLM:
        note = (
            f"\n(Note: CSP found {len(possible)} words; "
            f"only {MAX_CANDIDATES_TO_LLM} were sent to the LLM.)\n"
        )

    return (
        f"ADDED ATTEMPT: {guess} -> {feedback}\n"
        f"POSSIBLE WORDS ({len(possible)}):\n{shown}\n"
        f"{note}\n"
        f"LLM DECISION:\n{content}"
    )

