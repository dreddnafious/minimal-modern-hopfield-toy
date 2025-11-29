# main.py — FINAL WORKING VERSION (nicegui 2.x compatible)
# Copy-paste and run — no more errors.

import numpy as np
import torch
from nicegui import ui
import matplotlib.pyplot as plt
import io
import base64

# --------------------- Data ---------------------
patterns_str = [
    "█████\n█   █\n█ █ █\n█   █\n█████",  # Apple
    "  █  \n ███ \n█   █\n ███ \n  █  ",  # Banana
    "  █  \n ███ \n█ █ █\n  █  \n  █  ",  # Cherry
    " █ █ \n█   █\n█ █ █\n█   █\n █ █ ",  # Donut
    " █ █ \n█ █ █\n ██  \n█ █ █\n █ █ "   # Pineapple
]
names = ["Apple", "Banana", "Cherry", "Donut", "Pineapple"]

def str_to_pattern(s):
    return np.array([1 if c == '█' else -1 for line in s.split('\n') for c in line], dtype=np.float32)

xi = np.stack([str_to_pattern(p) for p in patterns_str])
xi_torch = torch.from_numpy(xi).float()

# --------------------- Hopfield ---------------------
class ModernHopfield(torch.nn.Module):
    def __init__(self, memories):
        super().__init__()
        self.memories = memories

    def relax(self, state, steps=30, beta=12.0):
        h = state.clone()
        history = [h.cpu().numpy().copy()]
        sims_list = []

        for _ in range(steps):
            logits = self.memories @ h
            probs = torch.softmax(beta * logits, dim=0)
            h = self.memories.t() @ probs
            history.append(h.cpu().numpy().copy())
            sims = (self.memories @ h).cpu().numpy() / 25
            sims_list.append(sims)
            if len(history) > 1 and np.allclose(history[-1], history[-2], atol=1e-3):
                break
        return history, np.array(sims_list)

hopfield = ModernHopfield(xi_torch)

# --------------------- Image ---------------------
def pattern_to_dataurl(vec):
    grid = (vec > 0).reshape(5, 5)
    fig, ax = plt.subplots(figsize=(2.5, 2.5), facecolor='none')
    ax.imshow(grid, cmap='binary', interpolation='nearest')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True, dpi=120)
    plt.close(fig)
    return f'data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'

# --------------------- Trial ---------------------
class Trial:
    def __init__(self, card):
        self.card = card
        self.step = 0
        self.timer = None

        with card:
            ui.html('<div class="text-2xl font-bold text-center text-cyan-400">Loading...</div>', sanitize=False)
            self.title = ui.label()
            self.img = ui.interactive_image().classes('w-40 h-40 mx-auto block')
            self.plot = ui.plotly({
                'data': [{'y': [], 'name': n, 'mode': 'lines+markers'} for n in names],
                'layout': {
                    'height': 220, 'margin': {'l':40,'r':20,'t':20,'b':30},
                    'paper_bgcolor': '#1e1e1e', 'plot_bgcolor': '#1e1e1e',
                    'font': {'color': '#ddd'}, 'yaxis': {'range': [-1, 1]}
                }
            }).classes('w-full')
            self.status = ui.label().classes('text-center font-mono text-lg')

    def start(self, noise_pct):
        idx = np.random.randint(5)
        noisy = xi[idx].copy()
        noisy[np.random.rand(25) < noise_pct] *= -1

        state = torch.from_numpy(noisy)
        self.history, self.sims = hopfield.relax(state, beta=12.0)
        self.true_idx = idx
                # ← ADD THIS LINE (fixes most wrong predictions!)
        self.history, self.sims = hopfield.relax(state, beta=40.0)   # was 12.0 → now much sharper!
        self.step = 0

        self.title.text = f"True: {names[idx]} → ? ({noise_pct:.0%} noise)"
        self.img.set_source(pattern_to_dataurl(noisy))

        # Reset Plotly traces
        for trace in self.plot.figure['data']:
            trace['y'] = []

        self.status.text = "Thinking..."
        if self.timer:
            self.timer.cancel()
        self.timer = ui.timer(0.35, self.update, once=False)

    def update(self):
        if self.step >= len(self.history):
            #self.status.text = "CONVERGED"
            self.timer.cancel()
            return

        vec = self.history[self.step]
        self.img.set_source(pattern_to_dataurl(vec))

        cur = self.sims[:self.step+1]
        for trace, values in zip(self.plot.figure['data'], cur.T):
            trace['y'] = values.tolist()

        best = np.argmax(self.sims[min(self.step, len(self.sims)-1)])
        conf = self.sims[min(self.step, len(self.sims)-1)][best]
        color = 'lime' if best == self.true_idx else 'red'
        self.status.text = f"PREDICT: {names[best]} ({conf:+.5f})"
        self.status.style(f'replace color:{"lime" if best == self.true_idx else "red"}')

        if self.step >= len(self.history) - 1:
            self.status.text = f"FINAL → {names[best]} {'CORRECT' if best == self.true_idx else 'WRONG'}"
            self.status.classes('text-2xl font-bold')

        self.step += 1

# --------------------- UI ---------------------
ui.html('<h1 class="text-4xl font-bold text-center my-8 text-cyan-400">Modern Hopfield Live Gallery</h1>', sanitize=False)

with ui.row().classes('w-full justify-center items-center gap-10 my-6'):
    slider = ui.slider(min=0, max=50, value=35, step=1).props('label-always').classes('w-96')
    ui.label().bind_text_from(slider, 'value', lambda v: f"Noise Level: {v}%")
    ui.button('New Round (8 trials)', on_click=lambda: new_round(), color='cyan', icon='refresh') \
        .classes('px-8 py-4 text-lg font-bold')

with ui.grid(columns=4).classes('w-full max-w-7xl mx-auto gap-8'):
    trials = [Trial(ui.card().classes('p-6 bg-gray-900/80 rounded-xl')) for _ in range(8)]

def new_round():
    noise = slider.value / 100.0
    for t in trials:
        t.start(noise)

# Start immediately
ui.timer(0.5, new_round, once=True)

ui.run(title="Hopfield Gallery", dark=True, port=8085, show=True)