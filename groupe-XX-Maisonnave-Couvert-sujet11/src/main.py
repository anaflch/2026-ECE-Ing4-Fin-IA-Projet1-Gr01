import pygame
import sys
from game_engine import Minesweeper
from gui import GameGUI
from csp_solver import CSPSolver

def main():
    pygame.init()
    AI_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(AI_EVENT, 150) # Vitesse de l'IA

    def reset_game():
        """Fonction pour (re)démarrer une partie"""
        print("\n--- NOUVELLE PARTIE ---")
        # Tu peux changer width/height ici si tu veux
        g = Minesweeper(width=15, height=15, num_mines=30)
        gui = GameGUI(g)
        s = CSPSolver(g)
        return g, gui, s

    # Initialisation première partie
    game, gui, solver = reset_game()
    
    running = True
    game_over = False
    game_status = None # "VICTOIRE" ou "DÉFAITE"

    while running:
        # Vérification Victoire
        if not game_over and len(game.revealed) + len(game.grid) == game.width * game.height:
            print("🏆 VICTOIRE ! Tous les pièges ont été évités.")
            game_over = True
            game_status = "VICTOIRE"

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # --- GESTION DU RESTART (Touche R) ---
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game, gui, solver = reset_game()
                    game_over = False
                    game_status = None
            
            # --- GESTION DU CLIC MANUEL ---
            elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                 gui.handle_click(pygame.mouse.get_pos())
            
            # --- GESTION IA ---
            elif event.type == AI_EVENT and not game_over:
                safe, mines = solver.solve()
                
                # Appliquer les drapeaux
                for m in mines: game.flags.add(m)
                
                # Appliquer les révélations
                for s in safe:
                    if (s[0], s[1]) not in game.revealed:
                        if game.reveal(s[0], s[1]):
                            print(f"💀 GAME OVER ! L'IA a explosé en {s}")
                            game_over = True
                            game_status = "DÉFAITE"
                            # On révèle tout pour montrer les dégâts
                            game.revealed.update(game.grid)

        # Dessin (avec le statut pour le message de fin)
        gui.draw(game_status)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()