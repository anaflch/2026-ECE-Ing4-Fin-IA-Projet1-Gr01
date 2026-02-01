# --- IMPORTATIONS ---
import os
import unicodedata
import re

# --- CONFIGURATION ---
DOSSIER_BASE = os.path.dirname(os.path.abspath(__file__))
# Le fichier d'entrée se trouve dans le sous-dossier 'donne_reponse' (relatif à ce script)
FICHIER_ENTREE = os.path.join(DOSSIER_BASE, "donne_reponse", "ouestfrance.txt")
# Le fichier de sortie sera dans le même dossier
FICHIER_SORTIE = os.path.join(DOSSIER_BASE, "donne_reponse", "dico_definitions_organise.txt")

# --- CLASSE DE TRAITEMENT ---
class DefinitionFormatter:
    def __init__(self):
        # Structure : { longueur : { mot : [def1, def2, ...] } }
        self.data = {}

    def _clean_word(self, word):
        """Nettoie le mot (clé) : majuscules, sans accents, A-Z uniquement."""
        nfkd = unicodedata.normalize('NFKD', word)
        ascii_only = nfkd.encode('ASCII', 'ignore').decode('utf-8')
        return re.sub(r'[^A-Z]', '', ascii_only.upper())

    def _clean_definition(self, text):
        """Corrige certaines coquilles d'OCR connues dans le fichier source."""
        # 1. D'abord les cas spécifiques (pour éviter que "ELL" ne soit remplacé trop tôt)
        text = text.replace(" ELL L ", " ELLE IL ")
        text = text.replace(" ELL N ", " ELLE ON ")
        text = text.replace(" EU AIS ", " EUX MAIS ")
        text = text.replace(" FIL LLE ", " FILS ELLE ")
        
        # 2. Ensuite les cas génériques
        text = text.replace(" ELL ", " ELLE ")
        return text

    def process(self, input_path, output_path):
        """Fonction principale : lit le fichier source et génère le dictionnaire."""
        print(f"--- Lecture du fichier : {input_path} ---")
        
        if not os.path.exists(input_path):
            print(f"ERREUR : Le fichier {input_path} n'existe pas.")
            return

        # On essaie d'ouvrir en utf-8, sinon latin-1 si erreur d'encodage
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                self._read_lines(f)
        except UnicodeDecodeError:
            print("Encodage UTF-8 échoué, tentative en Latin-1...")
            with open(input_path, 'r', encoding='latin-1') as f:
                self._read_lines(f)

        print(f"Lecture terminée. Organisation des données...")
        
        # Ajout manuel des mots de 1 lettre pour faciliter la génération
        if 1 not in self.data:
            self.data[1] = {}
        self.data[1]['A'] = ['xxx']
        self.data[1]['Y'] = ['xxx']
        
        self._save(output_path)

    def _read_lines(self, file_obj):
        """Parcourt le fichier ligne par ligne pour extraire mots et définitions."""
        for line in file_obj:
            line = line.strip()
            if not line: continue
            
            # Séparation par tabulation (format du fichier ouestfrance)
            parts = line.split('\t')
            if not parts: continue
            
            raw_word = parts[0]
            definitions = parts[1:] # Tout le reste, ce sont des définitions
            
            # Nettoyage du mot clé
            word = self._clean_word(raw_word)
            if not word: continue
            
            length = len(word)
            
            if length not in self.data:
                self.data[length] = {}
            
            # Ajout des définitions
            if word not in self.data[length]:
                self.data[length][word] = []
            
            # On ajoute les nouvelles définitions en évitant les doublons
            for d in definitions:
                d = d.strip()
                d = self._clean_definition(d)
                if d and d not in self.data[length][word]:
                    self.data[length][word].append(d)

    def _save(self, output_path):
        """Écrit le résultat trié dans le fichier final."""
        print(f"--- Sauvegarde dans : {output_path} ---")
        
        # Création du dossier si nécessaire
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Tri par longueur (1, 2, 3...)
            for length in sorted(self.data.keys()):
                words_dict = self.data[length]
                f.write(f"--- LONGUEUR {length} ({len(words_dict)} mots) ---\n")
                
                # Tri alphabétique des mots
                for word in sorted(words_dict.keys()):
                    defs = words_dict[word]
                    # Format : MOT : ['Def1', 'Def2']
                    f.write(f"{word} : {defs}\n")
                    
        print("Sauvegarde réussie !")

# --- POINT D'ENTRÉE ---
if __name__ == "__main__":
    formatter = DefinitionFormatter()
    formatter.process(FICHIER_ENTREE, FICHIER_SORTIE)
