import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set dark theme for plots
plt.style.use('dark_background')

# ===================== DEVICE CONFIG =====================

# Auto-detect GPU or use CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ===================== CONFIG =====================

N_NEURONS = 500
GRID_H, GRID_W = 20, 25
assert GRID_H * GRID_W == N_NEURONS

TOTAL_STEPS = 10000

NUM_SPARKS = 3

STATE_DECAY = 0.95
NOISE_STD = 0.05

LR_EDGE = 0.05          # strengthen used edges
LR_GLOBAL_DECAY = 0.001 # small decay everywhere

# Memory system (inspired by v26)
MEM_DECAY = 0.92        # memory field fading per step
MEM_DEPOSIT = 0.15      # how much memory sparks deposit
MEM_BIAS = 0.8          # how strongly memory influences movement
EXPLORE_CHANCE = 0.05   # random exploration probability

# Movement control
TEMPERATURE = 0.3       # softmax temperature for movement

# Graph layout physics
LAYOUT_FORCE_STEPS = 5
LAYOUT_LR = 0.01
GRAPH_UPDATE_FREQ = 5  # Update graph every N steps

# Spark properties
SPARK_ENERGY_DECAY = 0.98
SPARK_MIN_ENERGY = 0.05
SPARK_FORCE_STEPS = 5
SATURATION_THRESHOLD = 0.99  # Avoid neurons with activation above this value


# ===================== FORCE-LAYOUT GRAPH =====================

def force_layout(W, steps=50, lr=0.01):
    """Simple force-directed graph layout for visualization."""
    n = W.shape[0]
    pos = np.random.randn(n, 2)

    # Convert weights to numpy safely
    Wp = np.clip(W.detach().cpu().numpy(), 0, None)
    if Wp.max() > 0:
        Wp /= Wp.max()

    for _ in range(steps):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2) + 1e-6

        # Repulsion force
        rep = diff / (dist[:, :, None] ** 2) * (-0.001)
        # Attraction force based on weights
        att = diff * Wp[:, :, None] * 0.02

        force = rep + att
        pos += lr * force.sum(axis=1)

    return pos


# ===================== NETWORK =====================

class SparkNetAlpha(nn.Module):
    def __init__(self, n, k, device=None):
        super().__init__()
        self.n = n
        self.k = k
        self.device = device if device is not None else DEVICE

        # Core network weights
        self.W = nn.Parameter(0.1 * torch.randn(n, n, device=self.device))

        # Neuron states
        self.register_buffer("s", torch.zeros(n, device=self.device))

        # Memory field - key innovation from v26
        self.register_buffer("M", torch.zeros(n, device=self.device))

        # Spark system
        self.spark_pos = torch.randint(0, n, (k,), device=self.device)
        self.spark_energy = torch.ones(k, device=self.device)
        self.spark_age = torch.zeros(k, dtype=torch.long, device=self.device)

        # Graph motion tracking - "Learning Temperature"
        self.prev_graph_pos = None
        self.motion_history = []

    def reset(self):
        self.s.zero_()
        self.M.zero_()
        self.spark_pos = torch.randint(0, self.n, (self.k,), device=self.device)
        self.spark_energy = torch.ones(self.k, device=self.device)
        self.spark_age = torch.zeros(self.k, dtype=torch.long, device=self.device)
        # Reset motion tracking
        self.prev_graph_pos = None
        self.motion_history = []

    def update_spark_count(self, new_k):
        """Update the number of sparks and reinitialize"""
        self.k = new_k
        self.spark_pos = torch.randint(0, self.n, (new_k,), device=self.device)
        self.spark_energy = torch.ones(new_k, device=self.device)
        self.spark_age = torch.zeros(new_k, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def step(self):
        # ===== 1) DECAY SYSTEMS =====
        self.s *= STATE_DECAY           # neuron activations fade
        self.M *= MEM_DECAY             # memory field fades

        # ===== 2) BASIC NETWORK UPDATE =====
        noise = NOISE_STD * torch.randn_like(self.s)
        self.s = torch.sigmoid(self.W @ self.s + noise)

        # ===== 3) SPARK PRESENCE =====
        # Young sparks force their neurons to high activation
        for i in range(self.k):
            if self.spark_age[i] < SPARK_FORCE_STEPS:
                self.s[self.spark_pos[i]] = 1.0

        # ===== 4) SPARK MOVEMENT WITH MEMORY BIAS =====
        for i in range(self.k):
            prev = int(self.spark_pos[i].item())

            # Get outgoing weights from current position
            row = self.W[prev, :]
            base_weights = torch.relu(row) + 1e-6

            # KEY INNOVATION: Combine weights with memory bias
            # Higher memory at target locations makes them more attractive
            logits = base_weights / TEMPERATURE + MEM_BIAS * self.M

            # SATURATION AVOIDANCE: Filter out over-activated neurons
            # This prevents sparks from repeatedly hitting the same hot spots
            saturation_mask = self.s < SATURATION_THRESHOLD
            logits = torch.where(saturation_mask, logits, torch.tensor(-1e9, device=self.device))

            # Sometimes ignore everything and explore randomly
            if torch.rand(1, device=self.device).item() < EXPLORE_CHANCE:
                next_pos = torch.randint(0, self.n, (1,), device=self.device).item()
            else:
                probs = torch.softmax(logits, dim=0)
                next_pos = torch.multinomial(probs, 1).item()

            self.spark_pos[i] = next_pos

            # ===== 5) UPDATE LONG-TERM STRUCTURE =====
            # Strengthen the connection that was just used
            self.W.data[next_pos, prev] = (
                self.W.data[next_pos, prev] * (1 - LR_EDGE)
                + self.s[prev] * LR_EDGE
            )

            # ===== 6) UPDATE MEMORY FIELD =====
            # Deposit memory at new location - this influences future movement
            self.M[next_pos] += MEM_DEPOSIT

            # ===== 7) SPARK LIFECYCLE =====
            self.spark_energy[i] *= SPARK_ENERGY_DECAY
            self.s[next_pos] = self.spark_energy[i]
            self.spark_age[i] += 1

            # Respawn spark if energy too low
            if self.spark_energy[i] < SPARK_MIN_ENERGY:
                # Respawn with memory bias - hotspots more likely
                if self.M.sum() > 0:
                    mem_probs = torch.softmax(self.M * 2.0, dim=0)  # amplify memory differences
                    new_pos = torch.multinomial(mem_probs, 1).item()
                else:
                    new_pos = torch.randint(0, self.n, (1,), device=self.device).item()

                self.spark_pos[i] = new_pos
                self.spark_energy[i] = 1.0
                self.spark_age[i] = 0

        # ===== 8) GLOBAL WEIGHT DECAY =====
        self.W.data *= (1 - LR_GLOBAL_DECAY)
        self.W.data.clamp_(-1, 1)

        return self.k


# ===================== PARAMETER INFO =====================

PARAM_INFO = {
    'TOTAL_STEPS': 'Total number of simulation steps to run',
    'NUM_SPARKS': 'Number of moving spark agents in the network',
    'STATE_DECAY': 'How fast neuron activations fade (0-1). Higher = slower fade, longer memory',
    'NOISE_STD': 'Random noise amplitude added each step. Higher = more chaos',
    'LR_EDGE': 'Learning rate for strengthening connections. Higher = faster pathway formation',
    'LR_GLOBAL_DECAY': 'Global weight decay rate. Higher = connections weaken faster',
    'MEM_DECAY': 'How fast memory field fades (0-1). Higher = memory lasts longer',
    'MEM_DEPOSIT': 'How much memory sparks deposit when passing. Higher = stronger trails',
    'MEM_BIAS': 'How strongly memory attracts sparks. Higher = prefer visited areas',
    'EXPLORE_CHANCE': 'Probability of random jumps (0-1). Higher = more random exploration',
    'TEMPERATURE': 'Movement randomness (0-1). Lower = more deterministic paths',
    'SPARK_ENERGY_DECAY': 'Energy loss per step (0-1). Lower = sparks die faster',
    'SPARK_MIN_ENERGY': 'Energy threshold for respawn. Higher = respawn sooner',
    'SPARK_FORCE_STEPS': 'Steps a new spark forces full activation. Higher = stronger initial pulse',
    'SATURATION_THRESHOLD': 'Activation level to avoid (0-1). Sparks skip neurons above this to prevent hot spots',
    'LAYOUT_FORCE_STEPS': 'Physics iterations for graph layout. Lower = faster but less stable',
    'LAYOUT_LR': 'Graph layout movement speed. Higher = faster settling but more jittery',
    'GRAPH_UPDATE_FREQ': 'Update graph every N steps. Lower = smoother animation but more CPU load'
}

# ===================== PLOTTING SETUP =====================

def setup_plots():
    # Create main figure with space for controls
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#2e2e2e')

    # Main visualization area (left side)
    gs_main = fig.add_gridspec(2, 2, left=0.05, right=0.6, hspace=0.3, wspace=0.25)

    ax_s = fig.add_subplot(gs_main[0, 0])  # Neuron activations
    ax_w = fig.add_subplot(gs_main[0, 1])  # Weight matrix
    ax_m = fig.add_subplot(gs_main[1, 0])  # Memory field
    ax_g = fig.add_subplot(gs_main[1, 1])  # Graph layout

    # Neuron activations plot
    s_img = ax_s.imshow(np.zeros((GRID_H, GRID_W)), vmin=0, vmax=1,
                        interpolation="nearest", cmap="viridis")
    ax_s.set_title("Neuron Activations", color='white', fontsize=12)
    ax_s.tick_params(colors='white')

    # Weight matrix plot
    w_img = ax_w.imshow(np.zeros((N_NEURONS, N_NEURONS)), cmap="bwr",
                        vmin=-0.3, vmax=0.3, interpolation="nearest", aspect='auto')
    ax_w.set_title("Weight Matrix W", color='white', fontsize=12)
    ax_w.tick_params(colors='white')
    cbar_w = plt.colorbar(w_img, ax=ax_w)
    cbar_w.ax.tick_params(colors='white')

    # Memory field plot
    m_img = ax_m.imshow(np.zeros((GRID_H, GRID_W)), vmin=0, vmax=1,
                        interpolation="nearest", cmap="plasma")
    ax_m.set_title("Memory Field M", color='white', fontsize=12)
    ax_m.tick_params(colors='white')
    cbar_m = plt.colorbar(m_img, ax=ax_m)
    cbar_m.ax.tick_params(colors='white')

    # Graph layout plot
    ax_g.set_title("2D Graph Layout (force physics)", color='white', fontsize=12)
    ax_g.set_xlim(-3, 3)
    ax_g.set_ylim(-3, 3)
    ax_g.tick_params(colors='white')
    ax_g.set_facecolor('#1e1e1e')
    g_scatter = ax_g.scatter([], [], s=6, c='cyan', alpha=0.7)

    # Control panel (right side)
    controls = setup_controls(fig)

    return fig, s_img, w_img, m_img, ax_g, g_scatter, controls


def setup_controls(fig):
    """Setup control buttons and settings panel"""
    controls = {}

    # Control buttons at top right
    btn_y = 0.95
    btn_height = 0.04
    btn_width = 0.07

    # Play button
    ax_play = fig.add_axes([0.65, btn_y, btn_width, btn_height])
    ax_play.set_facecolor('#4CAF50')
    btn_play = mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.1",
                                       facecolor='#4CAF50', edgecolor='white', linewidth=2)
    ax_play.add_patch(btn_play)
    ax_play.text(0.5, 0.5, 'PLAY', ha='center', va='center',
                 color='white', fontsize=11, weight='bold')
    ax_play.set_xlim(0, 1)
    ax_play.set_ylim(0, 1)
    ax_play.axis('off')
    controls['btn_play'] = (ax_play, 'play')

    # Pause button
    ax_pause = fig.add_axes([0.73, btn_y, btn_width, btn_height])
    ax_pause.set_facecolor('#FF9800')
    btn_pause = mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.1",
                                        facecolor='#FF9800', edgecolor='white', linewidth=2)
    ax_pause.add_patch(btn_pause)
    ax_pause.text(0.5, 0.5, 'PAUSE', ha='center', va='center',
                  color='white', fontsize=11, weight='bold')
    ax_pause.set_xlim(0, 1)
    ax_pause.set_ylim(0, 1)
    ax_pause.axis('off')
    controls['btn_pause'] = (ax_pause, 'pause')

    # Stop button
    ax_stop = fig.add_axes([0.81, btn_y, btn_width, btn_height])
    ax_stop.set_facecolor('#F44336')
    btn_stop = mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.1",
                                       facecolor='#F44336', edgecolor='white', linewidth=2)
    ax_stop.add_patch(btn_stop)
    ax_stop.text(0.5, 0.5, 'STOP', ha='center', va='center',
                 color='white', fontsize=11, weight='bold')
    ax_stop.set_xlim(0, 1)
    ax_stop.set_ylim(0, 1)
    ax_stop.axis('off')
    controls['btn_stop'] = (ax_stop, 'stop')

    # Save button (for parameter changes)
    ax_save = fig.add_axes([0.89, btn_y, btn_width, btn_height])
    ax_save.set_facecolor('#9C27B0')
    btn_save = mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.1",
                                       facecolor='#9C27B0', edgecolor='white', linewidth=2)
    ax_save.add_patch(btn_save)
    ax_save.text(0.5, 0.5, 'SAVE', ha='center', va='center',
                 color='white', fontsize=11, weight='bold')
    ax_save.set_xlim(0, 1)
    ax_save.set_ylim(0, 1)
    ax_save.axis('off')
    controls['btn_save'] = (ax_save, 'save')

    # Settings panel - more compact
    settings_params = [
        ('TOTAL_STEPS', TOTAL_STEPS),
        ('NUM_SPARKS', NUM_SPARKS),
        ('STATE_DECAY', STATE_DECAY),
        ('NOISE_STD', NOISE_STD),
        ('LR_EDGE', LR_EDGE),
        ('LR_GLOBAL_DECAY', LR_GLOBAL_DECAY),
        ('MEM_DECAY', MEM_DECAY),
        ('MEM_DEPOSIT', MEM_DEPOSIT),
        ('MEM_BIAS', MEM_BIAS),
        ('EXPLORE_CHANCE', EXPLORE_CHANCE),
        ('TEMPERATURE', TEMPERATURE),
        ('SPARK_ENERGY_DECAY', SPARK_ENERGY_DECAY),
        ('SPARK_MIN_ENERGY', SPARK_MIN_ENERGY),
        ('SPARK_FORCE_STEPS', SPARK_FORCE_STEPS),
        ('SATURATION_THRESHOLD', SATURATION_THRESHOLD),
        ('LAYOUT_FORCE_STEPS', LAYOUT_FORCE_STEPS),
        ('LAYOUT_LR', LAYOUT_LR),
        ('GRAPH_UPDATE_FREQ', GRAPH_UPDATE_FREQ),
    ]

    y_start = 0.88
    y_step = 0.017  # Compact with slight spacing
    controls['params'] = {}
    controls['info_boxes'] = {}
    controls['input_boxes'] = {}

    for i, (param_name, param_value) in enumerate(settings_params):
        y_pos = y_start - i * y_step

        # Parameter label (shorter)
        ax_label = fig.add_axes([0.65, y_pos, 0.12, 0.015])
        ax_label.text(0, 0.5, param_name, ha='left', va='center',
                      color='white', fontsize=7)
        ax_label.axis('off')

        # Value input box (editable background) - wider for numbers
        ax_value = fig.add_axes([0.78, y_pos, 0.12, 0.015])
        rect = Rectangle((0, 0), 1, 1, facecolor='#3e3e3e', edgecolor='#666666', linewidth=1)
        ax_value.add_patch(rect)
        value_text = ax_value.text(0.5, 0.5, f'{param_value}', ha='center', va='center',
                                   color='#00ff00', fontsize=7, family='monospace')
        ax_value.set_xlim(0, 1)
        ax_value.set_ylim(0, 1)
        ax_value.axis('off')
        controls['params'][param_name] = (ax_value, value_text, param_value)
        controls['input_boxes'][param_name] = ax_value

        # Info button (smaller)
        ax_info = fig.add_axes([0.91, y_pos, 0.015, 0.015])
        info_circle = mpatches.Circle((0.5, 0.5), 0.35, facecolor='#2196F3',
                                      edgecolor='white', linewidth=1)
        ax_info.add_patch(info_circle)
        ax_info.text(0.5, 0.5, 'i', ha='center', va='center',
                     color='white', fontsize=8, weight='bold', style='italic')
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis('off')
        controls['info_boxes'][param_name] = ax_info

    # Info popup area
    ax_popup = fig.add_axes([0.65, 0.05, 0.31, 0.12])
    ax_popup.set_facecolor('#1a1a1a')
    ax_popup.add_patch(Rectangle((0, 0), 1, 1, facecolor='#1a1a1a',
                                 edgecolor='#2196F3', linewidth=2))
    popup_text = ax_popup.text(0.5, 0.5, '', ha='center', va='center',
                               color='#ffffff', fontsize=7, wrap=True)
    ax_popup.set_xlim(0, 1)
    ax_popup.set_ylim(0, 1)
    ax_popup.axis('off')
    ax_popup.set_visible(False)
    controls['popup'] = (ax_popup, popup_text)

    # Motion plot (aligned with main visualizations bottom) - "Learning Temperature"
    ax_motion = fig.add_axes([0.65, 0.23, 0.32, 0.15])
    ax_motion.set_facecolor('#1e1e1e')
    ax_motion.set_title('Graph Motion (Learning Temperature)', color='white', fontsize=9)
    ax_motion.set_xlabel('Step', color='white', fontsize=8)
    ax_motion.set_ylabel('Motion', color='white', fontsize=8)
    ax_motion.tick_params(colors='white', labelsize=7)
    motion_line, = ax_motion.plot([], [], color='cyan', linewidth=1.5)
    ax_motion.grid(True, alpha=0.2, color='white')
    controls['motion'] = (ax_motion, motion_line)

    return controls


# ===================== MAIN LOOP =====================

class SimulationState:
    """Manages simulation state for play/pause/stop controls"""
    def __init__(self):
        self.running = False  # Start paused
        self.step = 0
        self.editing_param = None
        self.param_values = {}  # Store current parameter values
        self.cursor_visible = True  # For blinking cursor
        self.cursor_blink_count = 0  # Counter for cursor blink timing

    def toggle_pause(self):
        self.running = not self.running

    def stop(self):
        self.running = False
        self.step = 0

    def play(self):
        self.running = True


def on_click(event, state, controls, net):
    """Handle mouse clicks on buttons and info icons"""
    if event.inaxes is None:
        # Click outside - hide popup
        ax_popup, _ = controls['popup']
        ax_popup.set_visible(False)
        return

    # Check control buttons
    for _, (ax, action) in [(k, v) for k, v in controls.items() if k.startswith('btn_')]:
        if event.inaxes == ax:
            if action == 'play':
                if not state.running:
                    state.play()
                    print("▶ Simulation started/resumed")
            elif action == 'pause':
                state.toggle_pause()
                status = "PAUSED" if not state.running else "RESUMED"
                print(f"⏸ Simulation {status}")
            elif action == 'stop':
                state.stop()
                print("⏹ Simulation stopped and reset")
            elif action == 'save':
                if not state.running:
                    apply_parameter_changes(state, controls, net)
                else:
                    print("⚠ Cannot save parameters while running. Pause or stop first.")
            return

    # Check info buttons
    for param_name, ax_info in controls['info_boxes'].items():
        if event.inaxes == ax_info:
            show_info_popup(param_name, controls)
            return

    # Check parameter input boxes for editing
    for param_name, ax_input in controls['input_boxes'].items():
        if event.inaxes == ax_input:
            if not state.running:
                state.editing_param = param_name
                # Update visual cursor
                update_editing_cursor(controls, param_name)
                print(f"✏ Editing {param_name}. Type new value and press SAVE.")
            else:
                print("⚠ Cannot edit parameters while running. Pause or stop first.")
            return


def update_editing_cursor(controls, param_name, show_cursor=True):
    """Update visual cursor indicator for editing"""
    # Clear previous highlights
    for pname, (ax_value, value_text, _) in controls['params'].items():
        if pname == param_name:
            # Highlight the editing field
            ax_value.set_facecolor('#555555')
            value_text.set_color('#ffff00')  # Yellow for editing

            # Add blinking cursor
            if show_cursor:
                current_text = value_text.get_text()
                value_text.set_text(current_text + '|')
            else:
                current_text = value_text.get_text()
                if current_text.endswith('|'):
                    value_text.set_text(current_text[:-1])
        else:
            # Reset to normal
            ax_value.set_facecolor('#3e3e3e')
            value_text.set_color('#00ff00')


def on_key(event, state, controls):
    """Handle keyboard input for parameter editing"""
    if state.editing_param is None:
        return

    param_name = state.editing_param
    _, value_text, _ = controls['params'][param_name]

    # Remove cursor before processing
    current_str = value_text.get_text()
    if current_str.endswith('|'):
        current_str = current_str[:-1]

    if event.key == 'enter':
        # Finish editing
        old_param = state.editing_param
        state.editing_param = None
        # Reset visual cursor
        ax_value, value_text, _ = controls['params'][old_param]
        ax_value.set_facecolor('#3e3e3e')
        value_text.set_color('#00ff00')
        # Remove cursor if present
        final_text = value_text.get_text()
        if final_text.endswith('|'):
            value_text.set_text(final_text[:-1])
        return

    if event.key == 'backspace':
        # Remove last character
        if len(current_str) > 0:
            new_str = current_str[:-1]
            value_text.set_text(new_str)
            state.param_values[param_name] = new_str
        else:
            value_text.set_text('')
        # Reset cursor blink
        state.cursor_visible = True
        state.cursor_blink_count = 0

    elif event.key in '0123456789.-':
        # Add digit or decimal point
        new_str = current_str + event.key
        value_text.set_text(new_str)
        state.param_values[param_name] = new_str
        # Reset cursor blink
        state.cursor_visible = True
        state.cursor_blink_count = 0


def show_info_popup(param_name, controls):
    """Show parameter information in popup"""
    info = PARAM_INFO.get(param_name, "No information available")
    ax_popup, popup_text = controls['popup']

    # Wrap text manually for better fit
    max_chars = 55
    words = info.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars:
            current_line += word + " "
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.strip())

    wrapped_text = f"{param_name}\n" + "\n".join(lines)
    popup_text.set_text(wrapped_text)
    ax_popup.set_visible(True)


def apply_parameter_changes(state, controls, net):
    """Apply changed parameters and reset simulation"""
    global TOTAL_STEPS, NUM_SPARKS, STATE_DECAY, NOISE_STD, LR_EDGE, LR_GLOBAL_DECAY
    global MEM_DECAY, MEM_DEPOSIT, MEM_BIAS, EXPLORE_CHANCE, TEMPERATURE
    global SPARK_ENERGY_DECAY, SPARK_MIN_ENERGY, SPARK_FORCE_STEPS, SATURATION_THRESHOLD
    global LAYOUT_FORCE_STEPS, LAYOUT_LR, GRAPH_UPDATE_FREQ

    changed = False
    spark_count_changed = False

    # Apply any edited values
    for param_name in state.param_values:
        try:
            new_value_str = state.param_values[param_name]
            if '.' in new_value_str:
                new_value = float(new_value_str)
            else:
                new_value = int(new_value_str)

            # Update global variable
            if param_name == 'TOTAL_STEPS':
                TOTAL_STEPS = new_value
            elif param_name == 'NUM_SPARKS':
                # Validate: can't have more sparks than neurons
                if new_value > N_NEURONS:
                    print(f"⚠ NUM_SPARKS ({new_value}) > N_NEURONS ({N_NEURONS}). Capping to {N_NEURONS}.")
                    new_value = N_NEURONS
                NUM_SPARKS = new_value
                spark_count_changed = True
            elif param_name == 'STATE_DECAY':
                STATE_DECAY = new_value
            elif param_name == 'NOISE_STD':
                NOISE_STD = new_value
            elif param_name == 'LR_EDGE':
                LR_EDGE = new_value
            elif param_name == 'LR_GLOBAL_DECAY':
                LR_GLOBAL_DECAY = new_value
            elif param_name == 'MEM_DECAY':
                MEM_DECAY = new_value
            elif param_name == 'MEM_DEPOSIT':
                MEM_DEPOSIT = new_value
            elif param_name == 'MEM_BIAS':
                MEM_BIAS = new_value
            elif param_name == 'EXPLORE_CHANCE':
                EXPLORE_CHANCE = new_value
            elif param_name == 'TEMPERATURE':
                TEMPERATURE = new_value
            elif param_name == 'SPARK_ENERGY_DECAY':
                SPARK_ENERGY_DECAY = new_value
            elif param_name == 'SPARK_MIN_ENERGY':
                SPARK_MIN_ENERGY = new_value
            elif param_name == 'SPARK_FORCE_STEPS':
                SPARK_FORCE_STEPS = new_value
            elif param_name == 'SATURATION_THRESHOLD':
                SATURATION_THRESHOLD = new_value
            elif param_name == 'LAYOUT_FORCE_STEPS':
                LAYOUT_FORCE_STEPS = new_value
            elif param_name == 'LAYOUT_LR':
                LAYOUT_LR = new_value
            elif param_name == 'GRAPH_UPDATE_FREQ':
                GRAPH_UPDATE_FREQ = new_value

            changed = True
            print(f"✓ {param_name} = {new_value}")

            # Update visual display
            _, value_text, _ = controls['params'][param_name]
            value_text.set_text(str(new_value))

        except ValueError:
            print(f"✗ Invalid value for {param_name}: {state.param_values[param_name]}")

    if changed:
        # Reset simulation with new parameters
        state.step = 0

        # Update spark count if changed
        if spark_count_changed:
            net.update_spark_count(NUM_SPARKS)
            print(f"✓ Updated spark count to {NUM_SPARKS}")
        else:
            net.reset()

        state.param_values.clear()
        print("✓ Parameters saved. Simulation reset.")


def main():
    net = SparkNetAlpha(N_NEURONS, NUM_SPARKS)
    net.reset()

    fig, s_img, w_img, m_img, ax_g, _g_scatter, controls = setup_plots()
    state = SimulationState()

    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event',
                          lambda event: on_click(event, state, controls, net))
    fig.canvas.mpl_connect('key_press_event',
                          lambda event: on_key(event, state, controls))

    # Show the window initially
    plt.show(block=False)

    # Disable always-on-top and focus-stealing behavior
    try:
        manager = fig.canvas.manager
        if manager and hasattr(manager, 'window'):
            window = manager.window
            # For TkAgg backend
            if hasattr(window, 'attributes'):
                window.attributes('-topmost', 0)
                window.focus_set = lambda: None  # Disable focus stealing
            # For Qt backends
            elif hasattr(window, 'setWindowFlag'):
                try:
                    from PyQt5.QtCore import Qt
                    window.setWindowFlag(Qt.WindowStaysOnTopHint, False)
                    window.activateWindow = lambda: None  # Disable activation
                except ImportError:
                    try:
                        from PyQt6.QtCore import Qt
                        window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
                        window.activateWindow = lambda: None  # Disable activation
                    except ImportError:
                        pass
    except Exception:
        pass  # Backend doesn't support this

    print("\n" + "="*60)
    print("SPARKNET ALPHA - Interactive Simulation")
    print("="*60)
    print("Controls:")
    print("  PLAY   - Start/resume simulation")
    print("  PAUSE  - Pause/unpause simulation")
    print("  STOP   - Reset simulation to step 0")
    print("  SAVE   - Apply parameter changes and reset")
    print("  (i)    - Click info icons for parameter details")
    print("\nEditing:")
    print("  Click parameter values to edit (when paused/stopped)")
    print("  Type new value, then click SAVE")
    print("="*60 + "\n")

    # Start paused
    fig.suptitle(f"Step 0/{TOTAL_STEPS} [PAUSED] — Sparks: {net.k} — Press PLAY to start",
                fontsize=14, x=0.02, ha="left", color='white')

    while plt.fignum_exists(fig.number):
        # Handle cursor blinking when editing
        if state.editing_param:
            state.cursor_blink_count += 1
            # Blink every 30 frames (about 0.5 seconds)
            if state.cursor_blink_count >= 30:
                state.cursor_blink_count = 0
                state.cursor_visible = not state.cursor_visible
                update_editing_cursor(controls, state.editing_param, state.cursor_visible)
                fig.canvas.draw_idle()

        if state.running:
            state.step += 1
            if state.step > TOTAL_STEPS:
                print(f"\n✓ Simulation completed ({TOTAL_STEPS} steps)")
                state.running = False
                continue

            spark_count = net.step()

            # Update main title with spark count only
            status = "RUNNING" if state.running else "PAUSED"
            fig.suptitle(f"Step {state.step}/{TOTAL_STEPS} [{status}] — Sparks: {spark_count}",
                        fontsize=14, x=0.02, ha="left", color='white')

            # Update neuron activations
            s_img.set_data(net.s.view(GRID_H, GRID_W).cpu().numpy())

            # Update weight matrix
            w_img.set_data(net.W.detach().cpu().numpy())

            # Update memory field
            M_clamped = torch.clamp(net.M, min=0.0, max=1.0)
            m_img.set_data(M_clamped.view(GRID_H, GRID_W).cpu().numpy())

            # Update graph layout more frequently for visible motion
            if state.step % GRAPH_UPDATE_FREQ == 0:
                layout = force_layout(net.W, steps=LAYOUT_FORCE_STEPS, lr=LAYOUT_LR)

                # Track motion - "Learning Temperature"
                if net.prev_graph_pos is not None:
                    motion = np.sum(np.linalg.norm(layout - net.prev_graph_pos, axis=1))
                    net.motion_history.append(motion)

                    # Update motion plot
                    ax_motion, motion_line = controls['motion']
                    history_window = net.motion_history[-500:]  # Last 500 points
                    motion_line.set_data(range(len(history_window)), history_window)
                    ax_motion.relim()
                    ax_motion.autoscale_view()

                net.prev_graph_pos = layout.copy()

                ax_g.clear()
                ax_g.scatter(layout[:, 0], layout[:, 1], s=6, c='cyan', alpha=0.7)
                ax_g.set_title("2D Graph Layout (force physics)", color='white', fontsize=12)
                ax_g.set_xlim(-3, 3)
                ax_g.set_ylim(-3, 3)
                ax_g.tick_params(colors='white')
                ax_g.set_facecolor('#1e1e1e')

            # Update canvas without stealing focus
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        else:
            # Update title when paused/stopped
            if state.step == 0:
                net.reset()
                s_img.set_data(net.s.view(GRID_H, GRID_W).cpu().numpy())
                w_img.set_data(net.W.detach().cpu().numpy())
                m_img.set_data(net.M.view(GRID_H, GRID_W).cpu().numpy())
                fig.suptitle(f"Step 0/{TOTAL_STEPS} [STOPPED] — Sparks: {net.k} — Press PLAY to start",
                            fontsize=14, x=0.02, ha="left", color='white')
                state.step = -1  # Prevent repeated reset
                fig.canvas.draw_idle()

            # Process events without drawing when paused
            fig.canvas.flush_events()

        # Small sleep to prevent CPU overload
        time.sleep(0.01)  # Increased to 10ms for smoother cursor blink

    print("\nSimulation window closed.")


if __name__ == "__main__":
    main()
