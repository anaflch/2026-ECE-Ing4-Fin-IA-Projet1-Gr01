import sys

# ---------------------------------------------------------------------------
# Optional dependency: "keyboard"
# ---------------------------------------------------------------------------
# Le module `keyboard` permet de détecter des touches globalement (ESC, etc.).
# Problèmes fréquents :
#   - nécessite parfois des droits admin (Windows) ou des permissions (Linux)
#   - peut ne pas être disponible sur tous les environnements
#   - et surtout: avec input(), la détection "en live" est limitée (voir plus bas)
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except Exception:
    KEYBOARD_AVAILABLE = False

from llm_agent import interroger_agent_wordle, load_dictionary


def main():
    """
    Point d'entrée CLI pour piloter le solver Wordle (CSP + Ollama).

    Fonctionnement :
      - charge un dictionnaire de mots ("wordle.txt")
      - boucle d'interaction :
            l'utilisateur saisit une tentative (guess + feedback)
            l'agent :
              * parse l'entrée (regex ou LLM fallback)
              * ajoute la contrainte à l'historique
              * filtre les candidats via CSP
              * demande au LLM de proposer un ranking / next guess
    """
    print("--- Wordle Solver (Ollama + CSP) ---")
    print("Input format: GUESS FEEDBACK  (V=green, J=yellow, G=gray)")
    print("Examples: ORATE GVVJG   |   ORATE -> GVVJG")
    print("Quit: type 'quit' or press Ctrl+C.\n")

    # 1) Chargement du dictionnaire (domaine CSP)
    dictionary = load_dictionary("wordle.txt")
    if not dictionary:
        # Si le dictionnaire est vide, le solver ne peut pas fonctionner.
        print("Dictionary is empty. Please check 'wordle.txt'.")
        sys.exit(1)

    # 2) Historique des tentatives (contraintes) conservé pendant la session
    attempts = []

    # 3) Boucle interactive
    while True:
        try:
            user_text = input("Enter your attempt: ").strip()
        except (EOFError, KeyboardInterrupt):
            # EOF (Ctrl+D) ou interruption (Ctrl+C) : sortie propre
            print("\nShutting down... Goodbye!")
            sys.exit(0)

        if KEYBOARD_AVAILABLE and keyboard.is_pressed("esc"):
            print("Shutting down... Goodbye!")
            sys.exit(0)

        if not user_text:
            # Entrée vide : on redemande
            continue

        print("\nThinking...\n")

        try:
            # L'agent modifie `attempts` (il append la tentative validée).
            # Il renvoie une string prête à afficher.
            result = interroger_agent_wordle(user_text, dictionary, attempts)
            print(result)
        except Exception as e:
            # On catch pour éviter de casser la session CLI sur une erreur ponctuelle
            print(f"Error: {e}")

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
