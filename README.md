# LNL demos

Interactive demonstrations for the Statistical Models course. Each demo is a Streamlit page. It runs on your own machine, or on Streamlit Community Cloud behind a class passcode (see [Hosting for students](#hosting-for-students)).

## Quick start

1. Open a terminal: **Anaconda Prompt**, PowerShell, or the VS Code terminal.
2. Go to this folder:
   ```
   cd C:\Users\franc\repos\lnl-exercises\demos
   ```
3. Start the app:
   ```
   C:\Users\franc\.conda\envs\lnl-demos\python.exe -m streamlit run Home.py
   ```
   If conda is on your PATH, this is equivalent:
   ```
   conda activate lnl-demos
   streamlit run Home.py
   ```
4. Open **http://localhost:8501** in a browser. The browser does not open by itself, because `.streamlit/config.toml` sets `headless = true`.
5. Click **▶ OPEN DEMO** on a demo card.
6. To stop the app, go back to the terminal and press **Ctrl+C**.

Run the command from inside `demos/`. Running it from elsewhere breaks the `lib` imports and skips the theme in `.streamlit/config.toml`.

## Using a demo (0301 CI Coverage)

1. In the **SIMULATION PARAMETERS** panel at the top, set the true β values, σ, the sample size n, and the number of simulations. **MORE SETTINGS** holds corr(x₁, x₂), the seed, the confidence level, t or z, and the pause per simulation.
2. Click **▶ RUN** to draw samples until the total is reached. **+1** draws one sample at a time, and **RESET** clears the run.
3. Any click stops a running simulation. Changing a setting starts over with a new fixed X.
4. For class, set **pause per simulation** to 0.05–0.2 s so students can watch the tables update. Set it to 0 for a fast run.
5. To show under-coverage, choose **z** and set **n** to 6.

## First-time setup (new machine)

```
conda create -n lnl-demos python=3.11 -y
conda activate lnl-demos
cd C:\Users\franc\repos\lnl-exercises\demos
pip install -r requirements.txt
```

## Hosting for students

Streamlit Community Cloud deploys from a GitHub repository, so this folder is pushed on its own to a GitHub repo. The course repo stays on GitLab, and the solutions stay private.

1. On GitHub, create an empty repo, e.g. `lnl-demos` (public is fine, since no passcode is in the code).
2. Push only this folder:
   ```
   cd C:\Users\franc\repos\lnl-exercises
   git push https://github.com/likefrankie/lnl-demos.git "$(git subtree split --prefix demos):refs/heads/main"
   ```
   Commit `demos/` first: the split only sees committed files. Run the same command after each change to `demos/`. The app redeploys by itself.
3. At https://share.streamlit.io, sign in with GitHub and choose **Create app → Deploy a public app from GitHub**. Then set:
   - Repository `<you>/lnl-demos`, branch `main`, main file `Home.py`
   - App URL: e.g. `lnl-exercises-demo` → `https://lnl-exercises-demo.streamlit.app`
   - **Advanced settings**: Python 3.11, and under Secrets:
     ```
     passcode = "choose-a-code"
     ```
4. Share the URL and the passcode with the class. To change the passcode, edit the secret in the app's **Settings → Secrets**. Students who are already in stay in until they reload the page.

The gate (`lib/auth.py`) runs at the top of every page, so a direct page link asks for the code too. With no `passcode` secret set, as on a local run, the gate is off. To test it locally, put the same line in `.streamlit/secrets.toml`. That file is git-ignored.

## Troubleshooting

- **"Port 8501 is already in use".** The app is already running in another terminal, so open http://localhost:8501 there. Or start a second copy with `--server.port 8502`.
- **Edits to `lib/` don't show up.** Streamlit reloads the page files but not the `lib/` modules. Stop the app with Ctrl+C and start it again.
- **`ModuleNotFoundError: lib`.** You ran the command outside `demos/`. Use `cd` to go into `demos/` first.

## Layout

- `Home.py`: the landing page, which lists the demos.
- `pages/`: one file per demo. The prefix is the lecture number, so `0301` refers to `0001-03-01`.
- `lib/simulate.py`: model designs, error draws, OLS fits, and critical values. It uses the course notation: p independent variables and n − p − 1 degrees of freedom.
- `lib/plots.py`: shared Plotly figures.
- `lib/theme.py`: the terminal-style CSS and figure styling.
- `lib/auth.py`: the passcode gate for the hosted app.
- `.streamlit/config.toml`: the dark theme and the server settings (localhost only, headless).

## Demos

| ID | Demo | Shows |
|---|---|---|
| 0301 | Unbiasedness & CI coverage | With X and β fixed and a new ε drawn for each simulation, the average of the estimates approaches β, and about 95% of the intervals contain β. |
