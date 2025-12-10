import time
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import viser
from viser.extras import ViserUrdf
from robot_descriptions.loaders.yourdfpy import load_robot_description


class UR5Visualizer:
    def __init__(self):
        self.server = viser.ViserServer()
        self.server.gui.configure_theme(
            dark_mode=True,
            control_width="large"
        )
        self.urdf = load_robot_description("ur5_description")
        self.robot = ViserUrdf(self.server, urdf_or_path=self.urdf)
        self.is_playing = False
        self.trajectory = self.torques = self.errors = self.times = None
        self.current_idx = 0
        self.dt = 0.01
        self.speed = 1.0

        with self.server.gui.add_folder("Simulation Runs"):
            self.run_dropdown = self.server.gui.add_dropdown(
                label="Select Run",
                options=["<no runs available>"],
                initial_value="<no runs available>",
            )

            self.run_dropdown.on_update(lambda event: self.switch_run(event.target.value))

        # Store multiple runs
        self.runs = {}
        self.plot_handles = []

        with self.server.gui.add_folder("Control Gains"):
            self.kp_slider = self.server.gui.add_slider(
                "Kp", min=0, max=500, step=1, initial_value=100
            )
            self.kd_slider = self.server.gui.add_slider(
                "Kd", min=0, max=50, step=0.5, initial_value=10
            )

        with self.server.gui.add_folder("Playback"):
            self.play_btn = self.server.gui.add_button("Play")
            self.restart_btn = self.server.gui.add_button("Restart")

            self.play_btn.on_click(lambda _: self.toggle())
            self.restart_btn.on_click(lambda _: self.restart())

    def add_run(self, name, traj, torques, errors, times):
        """Add a simulation run to the visualizer."""
        self.runs[name] = (traj, torques, errors, times)
        self.run_dropdown.options = list(self.runs.keys())
        if len(self.runs) == 1:
            self.run_dropdown.value = name
            self.switch_run(name)

    def switch_run(self, name):
        """Switch to a different run."""
        if name not in self.runs:
            return
        traj, torques, errors, times = self.runs[name]
        self.trajectory, self.torques, self.errors, self.times = traj, torques, errors, times
        self.dt = times[1] - times[0] if len(times) > 1 else 0.01
        self.current_idx = 0
        self.robot.update_cfg(traj[0])
        self.update_plots(name)

    def seek(self, idx):
        if self.trajectory is not None and not self.is_playing:
            idx = max(0, min(idx, len(self.trajectory) - 1))
            self.current_idx = idx
            self.robot.update_cfg(self.trajectory[idx])

    def toggle(self):
        self.is_playing = not self.is_playing
        self.play_btn.name = "Pause" if self.is_playing else "Play"

    def restart(self):
        self.current_idx = 0
        if self.trajectory is not None:
            self.robot.update_cfg(self.trajectory[0])

    def update_plots(self, name):
        """Update live Plotly plots in GUI."""
        # Remove old plots
        for h in self.plot_handles:
            try:
                h.remove()
            except:
                pass
        self.plot_handles.clear()

        if self.trajectory is None:
            return

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # Joint Angles plot
        fig = go.Figure()
        for i in range(6):
            fig.add_trace(go.Scatter(
                x=self.times, y=self.trajectory[:, i],
                mode='lines', name=f'J{i+1}', line=dict(color=colors[i])
            ))
        fig.update_layout(
            title=f'{name}: Joint Angles', xaxis_title='Time (s)',
            yaxis_title='Angle (rad)', margin=dict(l=20, r=20, t=40, b=20)
        )
        self.plot_handles.append(self.server.gui.add_plotly(figure=fig, aspect=1.5))

        # Joint Torques plot
        fig = go.Figure()
        for i in range(6):
            fig.add_trace(go.Scatter(
                x=self.times, y=self.torques[:, i],
                mode='lines', name=f'J{i+1}', line=dict(color=colors[i])
            ))
        fig.update_layout(
            title=f'{name}: Joint Torques', xaxis_title='Time (s)',
            yaxis_title='Torque (Nm)', margin=dict(l=20, r=20, t=40, b=20)
        )
        self.plot_handles.append(self.server.gui.add_plotly(figure=fig, aspect=1.5))

        # Tracking Errors plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.times, y=self.errors[:, 0],
            mode='lines', name='Angular ||ωb||', line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=self.times, y=self.errors[:, 1],
            mode='lines', name='Linear ||vb||', line=dict(color='red')
        ))
        fig.update_layout(
            title=f'{name}: Tracking Errors', xaxis_title='Time (s)',
            yaxis_title='Error', margin=dict(l=20, r=20, t=40, b=20)
        )
        self.plot_handles.append(self.server.gui.add_plotly(figure=fig, aspect=1.5))

    def save(self, out: Path):
        """Save current run to folder with PNG plots and CSV data."""
        if self.trajectory is None:
            return
        out.mkdir(exist_ok=True)

        # Save PNG plots using matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(6):
            ax.plot(self.times, self.trajectory[:, i], label=f'Joint {i+1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (rad)')
        ax.set_title('Joint Angles vs Time')
        ax.legend()
        ax.grid()
        fig.savefig(out / 'joint_angles.png', dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(6):
            ax.plot(self.times, self.torques[:, i], label=f'Joint {i+1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Torque (Nm)')
        ax.set_title('Joint Torques vs Time')
        ax.legend()
        ax.grid()
        fig.savefig(out / 'joint_torques.png', dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.times, self.errors[:, 0], label='Angular ||ωb||')
        ax.plot(self.times, self.errors[:, 1], label='Linear ||vb||')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error')
        ax.set_title('Tracking Errors vs Time')
        ax.legend()
        ax.grid()
        fig.savefig(out / 'tracking_errors.png', dpi=150)
        plt.close()

        # Save CSV
        with open(out / 'simulation.csv', 'w', newline='') as f:
            csv.writer(f).writerows(self.trajectory)

        print(f"Saved to {out}")

    def run(self):
        print(f"Viser: {self.server.request_share_url()}")
        while True:
            if self.is_playing and self.trajectory is not None:
                if self.current_idx < len(self.trajectory):
                    time.sleep(self.dt / self.speed)
                    self.robot.update_cfg(self.trajectory[self.current_idx])
                    self.current_idx += 1
                    time.sleep(self.dt / self.speed)
                else:
                    self.is_playing = False
                    self.play_btn.name = "Play"

            time.sleep(0.01)
