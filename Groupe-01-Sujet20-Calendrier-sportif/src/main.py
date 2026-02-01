from typing import List, Dict, Tuple
from ortools.sat.python import cp_model  # Solveur Or tools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class TournamentScheduler:

    def __init__(self, n_teams):

        if n_teams % 2 != 0:
            n_teams += 1  # Rendre pair le nombre de team quoi qu'il arrive

        self.n_teams = n_teams
        self.teams = self.create_teams()
        self.rounds = n_teams - 1
        self.total_rounds = self.rounds * 2  # définiation du nombre de jours du tournoi

    def create_teams(self) -> List[str]:

        cities = ["Paris", "Lyon", "Marseille", "Lille", "Bordeaux", "Toulouse",
                  "Nice", "Nantes", "Strasbourg", "Montpellier", "Rennes", "Reims"] # nombre total de team (1 à 12)

        selected_cities = cities[:self.n_teams]
        return [f"{city}" for city in selected_cities] # Prendre le nombre nécessaire de villes

    def print_teams(self):
        print("\n" + "=" * 40)
        print("ÉQUIPES PARTICIPANTES")
        print("=" * 40)

        for i, team in enumerate(self.teams):
            print(f"Équipe {i}: {team}")
        print()

    def solve_tournament(self, max_breaks: int = 1) -> Dict[int, List[Tuple[int, int]]]:

        model = cp_model.CpModel()
        solver = cp_model.CpSolver()

        # Variables principales qui prend en compte toute les possibilité de match ( 1 si le match ce joue 0 sinon)
        match_vars = {}
        for i in range(self.n_teams):
            for j in range(self.n_teams):
                if i != j:
                    for r in range(self.total_rounds):
                        match_vars[(i, j, r)] = model.NewBoolVar(f"match_{i}_{j}_{r}")

        # Variable qui contient toute les décision des matches domicile ou extèrieur (1 si le match est à domicile sinon 0), il permet est résolution bein plus rapide dans le solveur
        is_home = {}
        for i in range(self.n_teams):
            for r in range(self.total_rounds):
                is_home[(i, r)] = model.NewBoolVar(f"is_home_{i}_{r}")

        # Variables de break
        break_vars = {}
        for i in range(self.n_teams):
            for r in range(self.total_rounds - 1):
                break_vars[(i, r)] = model.NewBoolVar(f"break_{i}_{r}")

        # CONTRAINTES

            # Chaque paire joue exactement 2 fois (aller-retour)
        for i in range(self.n_teams):
            for j in range(i + 1, self.n_teams):
                model.Add(
                    sum(
                        match_vars[(i, j, r)] + match_vars[(j, i, r)]
                        for r in range(self.total_rounds)
                    ) == 2
                )

            # Une équipe joue exactement un match par journée
        for i in range(self.n_teams):
            for r in range(self.total_rounds):
                model.Add(
                    sum(match_vars[(i, j, r)] for j in range(self.n_teams) if j != i) +
                    sum(match_vars[(j, i, r)] for j in range(self.n_teams) if j != i)
                    == 1
                )

            # Définition domicile / extérieur
        for i in range(self.n_teams):
            for r in range(self.total_rounds):
                model.Add(
                    sum(match_vars[(i, j, r)] for j in range(self.n_teams) if j != i)
                    == is_home[(i, r)]
                )

            # Équilibre domicile / extérieur
        for i in range(self.n_teams):
            model.Add(
                sum(is_home[(i, r)] for r in range(self.total_rounds))
                == self.rounds
            )

            # Détection correcte des breaks
        for i in range(self.n_teams):
            for r in range(self.total_rounds - 1):

                model.Add(
                    break_vars[(i, r)] >=
                    is_home[(i, r)] + is_home[(i, r + 1)] - 1
                )

                model.Add(
                    break_vars[(i, r)] >=
                    (1 - is_home[(i, r)]) + (1 - is_home[(i, r + 1)]) - 1
                )

        # Limiter le nombre total de breaks
        total_breaks = sum(break_vars.values())
        model.Add(total_breaks <= (max_breaks * self.n_teams)-2) # Appliquer comme contarinte la limite théorque du nombre de break (n-2 break)

        # Objectif : minimiser les breaks
        model.Minimize(total_breaks)

        #Résolution par le solveur
        solver.parameters.max_time_in_seconds = 30.0
        solver.parameters.num_search_workers = 10

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:  # Retourner la solution si le solveur trouve une solution optimal ou simplement faisable
            print("\nSolution trouvée")
            schedule = self.extract_schedule(solver, match_vars)
            return schedule
        else:
            print("Aucune solution trouvée")
            return {}

    def extract_schedule(self, solver, match_vars) -> Dict[int, List[Tuple[int, int]]]:  # Methode permettant de traduire toute les valeur booléenne de match_vars en donné utilisable (on ne garde que les valeur de match_vars = 1)

        schedule = {r: [] for r in range(self.total_rounds)}

        for (i, j, r), var in match_vars.items():
            if solver.Value(var) == 1:
                schedule[r].append((i, j))  # i reçoit j

        return schedule

    def print_schedule(self, schedule: Dict[int, List[Tuple[int, int]]]): # Affichage du calendrier dans le terminal
        print("\n")
        print("=" * 40)
        print("CALENDRIER DU TOURNOI")
        print("=" * 40)

        for r in sorted(schedule.keys()):
            print(f"\nJOURNÉE {r + 1}:")
            print("-" * 40)

            matches = schedule[r]
            for home_idx, away_idx in matches:
                home_team = self.teams[home_idx]
                away_team = self.teams[away_idx]
                print(f"  • {home_team:20} vs {away_team:20}")

        self.print_statistics(schedule)

    def print_statistics(self, schedule: Dict[int, List[Tuple[int, int]]]): #affichage des statistiques dans le terminal
        print("\n")
        print("=" * 40)
        print("STATISTIQUES DU CALENDRIER")
        print("=" * 40)

        stats = {i: {'home': 0, 'away': 0, 'breaks': 0} for i in range(self.n_teams)} # Compter les matchs par équipe et par type
        last_location = {i: None for i in range(self.n_teams)}

        for r in sorted(schedule.keys()): # Parcourir les journées dans l'ordre
            for home_idx, away_idx in schedule[r]:

                # Compter domicile/extérieur
                stats[home_idx]['home'] += 1
                stats[away_idx]['away'] += 1

                # Détection des breaks
                if last_location[home_idx] == 'home':
                    stats[home_idx]['breaks'] += 1
                if last_location[away_idx] == 'away':
                    stats[away_idx]['breaks'] += 1

                # Mettre à jour la dernière localisation
                last_location[home_idx] = 'home'
                last_location[away_idx] = 'away'

        print("\nÉQUILIBRE DOMICILE/EXTÉRIEUR:")
        for i in range(self.n_teams):
            team_stats = stats[i]
            diff = abs(team_stats['home'] - team_stats['away'])
            print(f"  {self.teams[i]:20}: {team_stats['home']:2}D / {team_stats['away']:2}E (diff: {diff})")

        print("\nBREAKS (matchs consécutifs au même endroit):")
        total_breaks = 0
        for i in range(self.n_teams):
            breaks = stats[i]['breaks']
            print(f"  {self.teams[i]:20}: {breaks} break{'s' if breaks > 1 else ''}")
            total_breaks += breaks

        print(f"\nTotal des breaks: {total_breaks}")
        print(f"Moyenne par équipe: {total_breaks / self.n_teams:.1f}")

    def visualize_schedule_gantt(self, schedule: Dict[int, List[Tuple[int, int]]]): # Affichage du calendirer sous la forme d'un diagramme de GANT
        fig, ax = plt.subplots(figsize=(14, 10))

        team_positions = {i: i for i in range(self.n_teams)}

        for r, matches in schedule.items():
            x_pos = r

            for home_idx, away_idx in matches:
                home_pos = team_positions[home_idx]
                away_pos = team_positions[away_idx]

                home_team = self.teams[home_idx]
                away_team = self.teams[away_idx]

                ax.barh(
                    home_pos - 0.2, 1, left=x_pos,
                    height=0.4, color='green', alpha=0.7,
                    label='Domicile' if r == 0 else ""
                )

                ax.text(
                    x_pos + 0.5, home_pos - 0.2,
                    away_team,
                    ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold'
                )

                ax.barh(
                    away_pos + 0.2, 1, left=x_pos,
                    height=0.4, color='red', alpha=0.7,
                    label='Extérieur' if r == 0 else ""
                )

                ax.text(
                    x_pos + 0.5, away_pos + 0.2,
                    home_team,
                    ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold'
                )

        # Configuration de l'axe Y
        ax.set_yticks(range(self.n_teams))
        ax.set_yticklabels(self.teams)
        ax.set_ylabel('Équipes', fontsize=15, fontweight='bold')

        # Configuration de l'axe X
        ax.set_xlabel('Journées', fontsize=15, fontweight='bold')
        ax.set_xlim(-0.2, len(schedule) + 0.2)
        ax.set_xlim(-0.2, len(schedule) + 0.2)
        ax.set_xticks(range(len(schedule)))

        x_labels = [f"J{r + 1}" for r in range(len(schedule))]
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

        # Légende
        home_patch = mpatches.Patch(color='green', alpha=0.7, label='Domicile')
        away_patch = mpatches.Patch(color='red', alpha=0.7, label='Extérieur')
        ax.legend(handles=[home_patch, away_patch], loc='upper right')

        ax.set_title(f"CALENDRIER DU CHAMPIONNAT - {self.n_teams} ÉQUIPES",
                     fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.show()


def main():
    print("\nCALENDRIER SPORTIF")

    print("\n1. INITIALISATION DU TOURNOI")

    # Choix du nombre d'équipe (4 à 12)
    scheduler = TournamentScheduler(n_teams=8)
    scheduler.print_teams()

    print("\n2. GÉNÉRATION DU CALENDRIER OPTIMISÉ")
    print("\nContraintes appliquées:")
    print("\n  • Chaque paire d'équipes joue 2 fois")
    print("  • Une équipe joue exactement 1 match par journée")
    print("  • Équilibre parfait domicile/extérieur")
    print("  • Minimisation des breaks")

    schedule = scheduler.solve_tournament(max_breaks=1)

    if schedule:
        scheduler.print_schedule(schedule)  # Afficher le calendrier complet

        scheduler.visualize_schedule_gantt(schedule) # Visualisation graphique


if __name__ == "__main__":
    main()

