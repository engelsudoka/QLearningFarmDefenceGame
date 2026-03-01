"""
FARM DEFENSE GAME WITH DEEP Q-LEARNING NETWORK (DQN)
=====================================================

GAME OVERVIEW:
--------------
A farm defense game where an AI agent (brown square / custom sprite) learns
to protect crops (green circles) from invading boars (black+red squares) by
shooting bullets.

REINFORCEMENT LEARNING COMPONENTS:
-----------------------------------
1. STATES  (8 continuous features, normalised to [0,1]):
   shooter_x, shooter_y, closest_boar_x, closest_boar_y,
   distance_to_boar, angle_to_boar, num_boars, ai_bullet_count

2. ACTIONS (10 discrete):
   0-7: move in 8 directions   |  8: Momentarily stay stationary |   9: shoot at nearest boar

3. REWARDS:
   +100  hit and eliminate a boar
     -5  bullet goes off-screen (miss)
     -1  every timestep (efficiency pressure)
    -10  boar reaches a vegetable

DQN ARCHITECTURE:
   Input(8) → FC(128,ReLU) → FC(64,ReLU) → Output(9) ~ 9792 trainable parameters

DQN ALGORITHM:
   Experience replay buffer (10 000 transitions)
   Separate target network (synced every 100 steps)
   Epsilon-greedy exploration, Adam optimiser (lr = 0.001)
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Requires Libraries and dependencis  
# ─────────────────────────────────────────────────────────────────────────────
import math
import os
import random
import sys
from collections import deque

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  ← edit these two paths to match your setup
# ─────────────────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1600, 1000
MODEL_PATH     = "dqn_model.pth"
AGENT_IMG_PATH = "agent.png"      # path to the AI agent sprite image  "./ship.bmp" 

# ─────────────────────────────────────────────────────────────────────────────
#  PYGAME INIT
# ─────────────────────────────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Farm Defense – DQN")
clock = pygame.time.Clock()

# ─────────────────────────────────────────────────────────────────────────────
#  COLOURS & FONTS
# ─────────────────────────────────────────────────────────────────────────────
BLACK        = (  0,   0,   0)
WHITE        = (255, 255, 255)
GRAY         = (128, 128, 128)
GREEN        = (  0, 200,   0)
DARK_GREEN   = ( 20,  60,  20)
YELLOW       = (255, 255,   0)
GOLD         = (255, 215,   0)
BROWN        = (139,  69,  19)
GOLDEN_BROWN = (153, 101,  21)
RED          = (220,  50,  50)
BLUE         = ( 30, 100, 255)
SKY_BLUE     = (135, 206, 235)
CYAN         = (  0, 220, 220)
ORANGE       = (255, 140,   0)

font          = pygame.font.Font(None, 24)
title_font    = pygame.font.Font(None, 36)
menu_font     = pygame.font.Font(None, 80)
menu_opt_font = pygame.font.Font(None, 52)

# ─────────────────────────────────────────────────────────────────────────────
#  OPTIONAL AGENT SPRITE: To Use your choice image for the Agent
# ─────────────────────────────────────────────────────────────────────────────
agent_image = None
try:
    if os.path.exists(AGENT_IMG_PATH):
        agent_image = pygame.transform.scale(
            pygame.image.load(AGENT_IMG_PATH).convert_alpha(), (50, 50))
        print(f"[INFO] Agent image loaded: {AGENT_IMG_PATH}")
    else:
        print(f"[INFO] '{AGENT_IMG_PATH}' not found – using default brown rectangle.")
except Exception as exc:
    print(f"[WARNING] Could not load agent image: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
#  STARTUP MENU: Tell keys to switch mode and exit game
# ═════════════════════════════════════════════════════════════════════════════

def show_menu() -> int:
    """
    Display the main menu.
    Returns: 1 = Play saved game  |  2 = Continue training  |  3 = Retrain.
    """
    model_exists = os.path.exists(MODEL_PATH)

    while True:
        screen.fill(DARK_GREEN)

        # Title
        t1 = menu_font.render("FARM DEFENSE", True, GOLD)
        t2 = menu_opt_font.render("Deep Q-Network Edition", True, WHITE)
        screen.blit(t1, (SCREEN_W // 2 - t1.get_width() // 2, 90))
        screen.blit(t2, (SCREEN_W // 2 - t2.get_width() // 2, 185))
        pygame.draw.line(screen, GOLD, (150, 260), (SCREEN_W - 150, 260), 2)

        # Options
        opts = [
            ("1   Play Saved Game",        GREEN  if model_exists else GRAY, model_exists),
            ("2   Continue Training",       YELLOW if model_exists else GRAY, model_exists),
            ("3   Retrain from Scratch",    ORANGE,                           True),
        ]
        mouse_pos    = pygame.mouse.get_pos()
        option_rects = []

        for i, (label, color, enabled) in enumerate(opts):
            y    = 305 + i * 130
            rect = pygame.Rect(SCREEN_W // 2 - 330, y - 15, 660, 74)
            option_rects.append((rect, i + 1, enabled))

            if enabled and rect.collidepoint(mouse_pos):
                pygame.draw.rect(screen, (50, 80, 50), rect, border_radius=10)

            surf = menu_opt_font.render(label, True, color)
            screen.blit(surf, (SCREEN_W // 2 - surf.get_width() // 2, y))

            if not enabled:
                note = font.render("(no saved model found)", True, (180, 180, 180))
                screen.blit(note, (SCREEN_W // 2 - note.get_width() // 2, y + 52))

        # Status & hint
        s_col = GREEN if model_exists else RED
        s_txt = f"Model: {'✓ found' if model_exists else '✗ not found'}   ({MODEL_PATH})"
        screen.blit(font.render(s_txt, True, s_col),
                    (SCREEN_W // 2 - font.size(s_txt)[0] // 2, SCREEN_H - 75))
        h_txt = "Press  1 / 2 / 3  or  click to select   •   ESC to quit"
        screen.blit(font.render(h_txt, True, WHITE),
                    (SCREEN_W // 2 - font.size(h_txt)[0] // 2, SCREEN_H - 45))

        pygame.display.flip()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_1 and model_exists: return 1
                if event.key == pygame.K_2 and model_exists: return 2
                if event.key == pygame.K_3:                  return 3
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for rect, choice, enabled in option_rects:
                    if rect.collidepoint(event.pos) and enabled:
                        return choice


# ═════════════════════════════════════════════════════════════════════════════
#  GAME STATE ENTITIES
# ═════════════════════════════════════════════════════════════════════════════

class Vegetable:
    """A crop on the farm that boars target."""
    def __init__(self):
        self.x = random.randint(300, SCREEN_W - 50)
        self.y = random.randint(50,  SCREEN_H - 50)


class Boar:
    """Invader that moves toward a vegetable target, spawning from screen edges."""
    def __init__(self, vegetables):
        edge = random.choice(["top", "bottom", "left", "right"])
        if   edge == "top":    self.x, self.y = random.randint(300, SCREEN_W), 0
        elif edge == "bottom": self.x, self.y = random.randint(300, SCREEN_W), SCREEN_H
        elif edge == "left":   self.x, self.y = 300,        random.randint(0, SCREEN_H)
        else:                  self.x, self.y = SCREEN_W, random.randint(0, SCREEN_H)
        self.speed  = 1.5
        self.size   = 50
        self.radius = 30
        self.target = random.choice(vegetables) if vegetables else None

    def move(self):
        if not self.target:
            return
        if   self.x < self.target.x: self.x += self.speed
        elif self.x > self.target.x: self.x -= self.speed
        if   self.y < self.target.y: self.y += self.speed
        elif self.y > self.target.y: self.y -= self.speed

    def reached_vegetable(self) -> bool:
        return bool(self.target and
                    math.hypot(self.x - self.target.x,
                               self.y - self.target.y) < 30)


class Bullet:
    """
    Projectile fired by either shooter.
    owner = "ai"    → drawn yellow, counts toward AI score
    owner = "human" → drawn cyan,   counts toward human score
    """
    def __init__(self, x: float, y: float, tx: float, ty: float,
                 owner: str = "ai"):
        self.x     = x
        self.y     = y
        self.speed = 12
        self.angle = math.atan2(ty - y, tx - x)
        self.owner = owner

    def move(self):
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)

    def off_screen(self) -> bool:
        return self.x < 300 or self.x > SCREEN_W or self.y < 0 or self.y > SCREEN_H


class Shooter:
    """AI-controlled agent (right side of the screen at episode start)."""
    def __init__(self):
        self.x     = SCREEN_W * 0.65
        self.y     = SCREEN_H * 0.50
        self.speed = 5
        self.size  = 50

    def move(self, dx: float, dy: float):
        self.x = max(0, min(SCREEN_W - self.size, self.x + dx))
        self.y = max(0, min(SCREEN_H - self.size, self.y + dy))

# ═════════════════════════════════════════════════════════════════════════════
#  DEEP Q-NETWORK Training
# ═════════════════════════════════════════════════════════════════════════════

class DQN(nn.Module):
    """
    Policy / target network.
    Input  : 8-feature state vector
    Output : Q-values for 9 discrete actions
    Hidden : 128 → 64 neurons, both with ReLU activation
    """
    def __init__(self, state_size: int = 8, action_size: int = 9):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Fixed-capacity circular experience replay buffer."""
    def __init__(self, max_size: int = 10000):
        self._buf = deque(maxlen=max_size)

    def add(self, s, a, r, ns, d):
        self._buf.append((s, a, r, ns, d))

    def sample(self, n: int):
        batch = random.sample(self._buf, min(n, len(self._buf)))
        s, a, r, ns, d = zip(*batch)
        return map(np.array, (s, a, r, ns, d))

    def __len__(self) -> int:
        return len(self._buf)


class DQNAgent:
    """
    DQN agent: experience replay + separate target network.
    The full training state (weights, optimiser, ε, global step) is saved
    to / loaded from MODEL_PATH so training can be resumed later.
    """
    def __init__(self):
        self.state_size  = 8
        self.action_size = 9

        self.policy_net = DQN(self.state_size, self.action_size)
        self.target_net = DQN(self.state_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.replay    = ReplayBuffer()

        self.epsilon            = 1.0
        self.epsilon_decay      = 0.995
        self.epsilon_min        = 0.01
        self.gamma              = 0.95
        self.batch_size         = 64
        self.target_update_freq = 100
        self.steps              = 0

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str = MODEL_PATH):
        torch.save(dict(
            policy  = self.policy_net.state_dict(),
            target  = self.target_net.state_dict(),
            optim   = self.optimizer.state_dict(),
            epsilon = self.epsilon,
            steps   = self.steps,
        ), path)
        print(f"[INFO] Model saved → {path}")

    def load(self, path: str = MODEL_PATH) -> bool:
        if not os.path.exists(path):
            print(f"[ERROR] Model not found: {path}")
            return False
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.policy_net.load_state_dict(ck["policy"])
        self.target_net.load_state_dict(ck["target"])
        self.optimizer.load_state_dict(ck["optim"])
        self.epsilon = ck.get("epsilon", 0.1)
        self.steps   = ck.get("steps",   0)
        print(f"[INFO] Model loaded ← {path}  "
              f"(ε={self.epsilon:.3f}, steps={self.steps:,})")
        return True

    # ── Kernel of the Reinforcement Learning (the core ────────────────────────────
    # ALL UPDATE IS IMPLEMENTED HERE

    def get_state(self, shooter: Shooter,
                  boars: list, bullets: list, _) -> np.ndarray:
        sx = shooter.x / SCREEN_W
        sy = shooter.y / SCREEN_H
        if boars:
            cb   = min(boars,
                       key=lambda b: math.hypot(b.x - shooter.x,
                                                b.y - shooter.y))
            bx   = cb.x / SCREEN_W
            by   = cb.y / SCREEN_H
            dx   = cb.x - shooter.x
            dy   = cb.y - shooter.y
            dist = math.hypot(dx, dy) / math.hypot(SCREEN_W, SCREEN_H)
            ang  = (math.atan2(dy, dx) + math.pi) / (2 * math.pi)
        else:
            bx = by = 0.5; dist = 1.0; ang = 0.5
        nb = len(boars) / 20
        bl = sum(1 for b in bullets if b.owner == "ai") / 10
        return np.array([sx, sy, bx, by, dist, ang, nb, bl], dtype=np.float32)

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            return int(self.policy_net(
                torch.FloatTensor(state).unsqueeze(0)).argmax())

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return
        s, a, r, ns, d = self.replay.sample(self.batch_size)
        s  = torch.FloatTensor(s)
        a  = torch.LongTensor(a)
        r  = torch.FloatTensor(r)
        ns = torch.FloatTensor(ns)
        d  = torch.FloatTensor(d)

        cur = self.policy_net(s).gather(1, a.unsqueeze(1))
        with torch.no_grad():
            nxt = self.target_net(ns).max(1)[0]
            tgt = r + (1 - d) * self.gamma * nxt

        loss = nn.MSELoss()(cur.squeeze(), tgt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)


# ═════════════════════════════════════════════════════════════════════════════
#  GAME LOGIC
# ═════════════════════════════════════════════════════════════════════════════

# Normalised direction vectors for the 8 movement actions
_DIRS = {
    0: (0, -1), 1: (0,  1), 2: (-1,  0), 3: (1,  0),
    4: (-1,-1), 5: (1, -1), 6: (-1,  1), 7: (1,  1),
}


def execute_action(action: int, shooter: Shooter,
                   boars: list, bullets: list):
    """Translate a DQN action index into a game-world effect."""
    if action in _DIRS:
        dx, dy = _DIRS[action]
        sp = shooter.speed * 2 #0.707
        shooter.move(dx * sp, dy * sp)
    elif action == 8 and boars:
        cb = min(boars, key=lambda b: math.hypot(b.x - shooter.x,
                                                  b.y - shooter.y))
        bullets.append(Bullet(
            shooter.x + shooter.size / 2, shooter.y + shooter.size / 2,
            cb.x      + cb.size      / 2, cb.y      + cb.size      / 2,
            owner="ai"))


def check_collisions(bullets: list, boars: list, vegetables: list):
    """
    Detect bullet↔boar and boar↔vegetable collisions.
    Modifies bullets and boars in-place.
    Returns (hits_ai, misses_ai, crops_damaged).
    """
    h_ai = m_ai = crops = 0

    for blt in bullets[:]:
        hit = False
        for boar in boars[:]:
            if (boar.x < blt.x < boar.x + boar.size and
                    boar.y < blt.y < boar.y + boar.size):
                bullets.remove(blt)
                boars.remove(boar)
                h_ai += 1
                hit = True
                break
        if not hit and blt.off_screen():
            bullets.remove(blt)
            m_ai += 1
            
    for boar in boars[:]:
        if boar.reached_vegetable():
            crops += 1
            boar.target = random.choice(vegetables) if vegetables else None

    return h_ai, m_ai, crops


# ═════════════════════════════════════════════════════════════════════════════
#  RENDERING
# ═════════════════════════════════════════════════════════════════════════════

_PANEL_W = 300


def draw_game(
        shooter: Shooter, boars: list, bullets: list, vegetables: list,
        ai_score: int,  ai_misses: int, episode: int, epsilon: float,
        crops_damaged: int, buf_size: int, mode: int):

    screen.fill((25, 80, 25))          # farm-green background

    # ── World ────────────────────────────────────────────────────────────
    for v in vegetables:
        pygame.draw.circle(screen, GREEN, (int(v.x), int(v.y)), 20)
        pygame.draw.circle(screen, BLACK, (int(v.x), int(v.y)), 20, 2)

    for b in boars:
        # Draw as gray circle (boar)
        pygame.draw.circle(screen, GRAY, (int(b.x), int(b.y)), b.radius)
        pygame.draw.circle(screen, (64, 64, 64), (int(b.x), int(b.y)), b.radius, 3)
        # Draw eyes
        pygame.draw.circle(screen, BLACK, (int(b.x - 7), int(b.y - 5)), 3)
        pygame.draw.circle(screen, BLACK, (int(b.x + 7), int(b.y - 5)), 3)
    for blt in bullets:
        col = GOLDEN_BROWN if blt.owner == "ai" else CYAN
        pygame.draw.ellipse(screen, col,
                            (int(blt.x) - 4, int(blt.y) - 9, 8, 18))

    # ── AI shooter ───────────────────────────────────────────────────────
    if agent_image:
        screen.blit(agent_image, (int(shooter.x), int(shooter.y)))
    else:
        pygame.draw.rect(screen, SKY_BLUE,
                         (int(shooter.x), int(shooter.y),
                          shooter.size, shooter.size))
        pygame.draw.rect(screen, WHITE,
                         (int(shooter.x), int(shooter.y),
                          shooter.size, shooter.size), 2)


    # ── HUD overlay ──────────────────────────────────────────────────────
    overlay = pygame.Surface((_PANEL_W, SCREEN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 168))
    screen.blit(overlay, (0, 0))

    MODE_LABEL = {1: "PLAY (saved)", 2: "TRAINING", 3: "TRAINING (new)"}
    MODE_COLOR = {1: CYAN,           2: YELLOW,     3: ORANGE}

    def hline(y_):
        pygame.draw.line(screen, GOLD, (8, y_), (_PANEL_W - 8, y_), 1)

    def row(text_, y_, color_=WHITE):
        screen.blit(font.render(text_, True, color_), (12, y_))

    # Header
    t = title_font.render("FARM DEFENSE", True, GOLD)
    screen.blit(t, (_PANEL_W // 2 - t.get_width() // 2, 8))
    hline(43)
    row(f"Mode : {MODE_LABEL.get(mode, '')}", 49, MODE_COLOR.get(mode, WHITE))
    hline(70)

    # AI stats
    row("── AI AGENT ──", 76, YELLOW)
    hline(94)
    for i, s in enumerate([
            f"Episode  : {episode}",
            f"Score    : {ai_score}",
            f"Misses   : {ai_misses}",
            f"Epsilon  : {epsilon:.3f}",
            f"Buffer   : {buf_size:,}"]):
        row(s, 100 + i * 22)
    hline(215)

    # Field stats
    row("── FIELD ──", 301, GREEN)
    hline(319)
    for i, s in enumerate([
            f"Boars    : {len(boars)}",
            f"Bullets  : {len(bullets)}",
            f"Crops hit: {crops_damaged}"]):
        row(s, 325 + i * 22)
    hline(394)

    # Controls
    row("── CONTROLS ──", 400, (200, 200, 200))
    for i, s in enumerate([
            "ESC        : Back to menu"]):
        row(s, 422 + i * 21, (185, 185, 185))

    pygame.display.flip()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    app_running = True

    while app_running:                             # Outer: menu → session → menu

        mode  = show_menu()
        agent = DQNAgent()
        training = mode in (2, 3)

        if mode == 1:                              # Play saved game
            if not agent.load():
                continue                           # Return to menu if load fails
            agent.epsilon = 0.0                    # Pure exploitation
            print("[INFO] Play mode – no training.")
        elif mode == 2:                            # Continue training
            if not agent.load():
                continue
            print(f"[INFO] Continue training  ε={agent.epsilon:.3f}")
        else:                                      # Retrain from scratch
            print("[INFO] Fresh training run.")

        vegetables    = [Vegetable() for _ in range(8)]

        episode         = 0
        hu_total_score  = 0
        hu_total_misses = 0

        if training:
            print(f"{'Ep':>6} {'AIScore':>8} {'Reward':>9} "
                  f"{'Eps':>7} {'BufSz':>7}")

        session_running = True
        while session_running and app_running:     # Middle: episodes

            episode += 1
            shooter = Shooter()
            boars   = [Boar(vegetables) for _ in range(5)]
            bullets: list = []

            ai_score = ai_misses = 0
            hu_score = hu_misses = 0
            crops_damaged  = 0
            episode_reward = 0.0
            step           = 0
            # Play mode: effectively unlimited steps per episode
            MAX_STEPS = 1_000 if training else 999_999_999

            state = agent.get_state(shooter, boars, bullets, vegetables)

            ep_running = True
            while ep_running and step < MAX_STEPS:  # Inner: steps

                # Events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        app_running = session_running = ep_running = False
                    if (event.type == pygame.KEYDOWN and
                            event.key == pygame.K_ESCAPE):
                        session_running = ep_running = False  # back to menu

                if not ep_running:
                    break

                # AI action
                action = agent.select_action(state)
                execute_action(action, shooter, boars, bullets)

                # Physics
                for blt  in bullets: blt.move()
                for boar in boars:   boar.move()

                # Collisions
                h_ai, m_ai, new_crops = check_collisions(
                    bullets, boars, vegetables)

                ai_score  += h_ai;  ai_misses  += m_ai
                crops_damaged += new_crops

                # Reward (AI agent only)
                reward = -1.0 + h_ai * 100 - m_ai * 5 - new_crops * 10
                episode_reward += reward

                # Keep boar count at 5
                while len(boars) < 5:
                    boars.append(Boar(vegetables))

                # Store transition and learn
                next_state = agent.get_state(shooter, boars, bullets, vegetables)
                if training:
                    done = (step >= MAX_STEPS - 1)
                    agent.replay.add(state, action, reward, next_state, done)
                    agent.train_step()

                state = next_state
                step += 1

                # Render  shooter,
                draw_game(
                    shooter, boars, bullets, vegetables,
                    ai_score, ai_misses, 
                    episode, agent.epsilon, crops_damaged,
                    len(agent.replay), mode)

                clock.tick(60)

            # ── End of episode ────────────────────────────────────────────
            if training and session_running:
                agent.decay_epsilon()
                print(f"{episode:6} {ai_score:8} {episode_reward:9.1f} "
                      f"{agent.epsilon:7.3f} {len(agent.replay):7}")
                if episode % 100 == 0:
                    agent.save()

        # ── End of session (ESC or window close) ─────────────────────────
        if training:
            agent.save()

    pygame.quit()


if __name__ == "__main__":
    main()