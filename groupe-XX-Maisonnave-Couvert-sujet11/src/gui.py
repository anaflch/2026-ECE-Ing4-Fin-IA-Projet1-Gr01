import pygame
import time

class GameGUI:
    def __init__(self, game, cell_size=40):
        self.game = game
        self.cell_size = cell_size
        self.width = game.width * cell_size
        self.height = game.height * cell_size
        
        # Configuration de la fenêtre
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Démineur IA - 'R' pour relancer")
        
        # Couleurs (Ton code original)
        self.COLORS = {
            'bg_hidden': (200, 200, 200),
            'bg_revealed': (255, 255, 255),
            'grid': (128, 128, 128),
            'flag': (255, 0, 0),
            'mine': (0, 0, 0),
            'text': {
                1: (0, 0, 255), 2: (0, 128, 0), 3: (255, 0, 0), 4: (0, 0, 128),
                5: (128, 0, 0), 6: (0, 128, 128), 7: (0, 0, 0), 8: (128, 128, 128)
            }
        }
        
        self.font = pygame.font.SysFont('Arial', 24, bold=True)
        # Nouvelle police pour le message de fin
        self.overlay_font = pygame.font.SysFont('Arial', 36, bold=True)

    def handle_click(self, pos):
        """Convertit le clic souris en coordonnées de grille"""
        x = pos[0] // self.cell_size
        y = pos[1] // self.cell_size
        if 0 <= x < self.game.width and 0 <= y < self.game.height:
            self.game.reveal(x, y)

    def draw_cell(self, x, y):
        """Dessine une case unique (Ton code original)"""
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
        val = self.game.get_value(x, y)
        
        if (x, y) in self.game.revealed:
            pygame.draw.rect(self.screen, self.COLORS['bg_revealed'], rect)
            if (x, y) in self.game.grid:
                pygame.draw.circle(self.screen, self.COLORS['mine'], rect.center, self.cell_size // 4)
            elif val > 0:
                text = self.font.render(str(val), True, self.COLORS['text'].get(val, (0,0,0)))
                self.screen.blit(text, text.get_rect(center=rect.center))
        else:
            pygame.draw.rect(self.screen, self.COLORS['bg_hidden'], rect)
            if (x, y) in self.game.flags:
                pygame.draw.rect(self.screen, self.COLORS['flag'], (rect.centerx - 5, rect.centery - 5, 10, 10))
        
        pygame.draw.rect(self.screen, self.COLORS['grid'], rect, 1)

    def draw_probabilities(self, prob_map):
        """Affiche un calque de couleur selon la probabilité (Ton code original)"""
        for (x, y), prob in prob_map.items():
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            intensity = int(255 * prob) 
            s = pygame.Surface((self.cell_size, self.cell_size))
            s.set_alpha(100)
            s.fill((intensity, 255 - intensity, 0))
            self.screen.blit(s, rect.topleft)
            
            if prob > 0.0:
                perc_text = pygame.font.SysFont('Arial', 10).render(f"{int(prob*100)}%", True, (0,0,0))
                self.screen.blit(perc_text, rect.topleft)

    def draw_restart_overlay(self, status):
        """(NOUVEAU) Affiche un message clignotant semi-transparent"""
        current_time = pygame.time.get_ticks()
        
        # Clignotement toutes les 800ms
        if (current_time // 800) % 2 == 0:
            return 

        text_str = f"{status} - Appuyez sur R"
        color = (0, 150, 0) if "VICTOIRE" in status else (200, 0, 0)
        
        text_surf = self.overlay_font.render(text_str, True, color)
        
        # Fond semi-transparent
        bg_surf = pygame.Surface((text_surf.get_width() + 20, text_surf.get_height() + 20))
        bg_surf.set_alpha(200)
        bg_surf.fill((255, 255, 255))
        
        center_x, center_y = self.width // 2, self.height // 2
        bg_rect = bg_surf.get_rect(center=(center_x, center_y))
        text_rect = text_surf.get_rect(center=(center_x, center_y))
        
        self.screen.blit(bg_surf, bg_rect)
        self.screen.blit(text_surf, text_rect)

    def draw(self, game_status=None):
        """Boucle de dessin principal mise à jour"""
        self.screen.fill((0, 0, 0))
        for x in range(self.game.width):
            for y in range(self.game.height):
                self.draw_cell(x, y)
        
        # Affichage des probabilités (Ton code original conservé)
        if hasattr(self.game, 'prob_map') and self.game.prob_map:
            self.draw_probabilities(self.game.prob_map)
            
        # Affichage du message de fin (Nouveau)
        if game_status:
            self.draw_restart_overlay(game_status)